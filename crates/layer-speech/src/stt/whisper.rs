//! Whisper, on the ONNX Runtime.
//!
//! The catalogue installs the int8-quantised exports: on a CPU shared with a
//! game and OBS they decode two to three times faster for a fraction of a
//! percent of word error rate. Nothing here depends on which one is on disk.
//!
//! Its encoder input is a fixed 30-second window, zero-padded, so a one-second
//! utterance costs as much to encode as a twenty-second one — which is why its
//! partials are treated as more expensive than Moonshine's.

use std::collections::HashSet;
use std::path::Path;

use ort::session::Session;
use serde_json::Value;

use super::features::WhisperFeatures;
use super::onnx::{Result, count_layers, create_session, err, f32_tensor, input_names, output_names, pick_name, read_json};
use super::seq2seq::{Generate, Greedy};
use super::tokenizer::{self, Detokenizer};

pub struct Whisper {
    encoder: Session,
    decoder: Session,
    detok: Detokenizer,
    features: WhisperFeatures,
    input_name: String,
    hidden_name: String,
    n_mels: usize,
    frames: usize,
    samples: usize,
    sample_rate: usize,
    greedy: Greedy,
    eos_id: usize,
    max_length: usize,
    start_ids: Vec<i64>,
    suppress: HashSet<usize>,
    begin_suppress: HashSet<usize>,
}

/// The first of several config fields that is set.
fn num(keys: &[(&Value, &str)], fallback: u64) -> u64 {
    keys.iter().find_map(|(obj, k)| obj[*k].as_u64()).unwrap_or(fallback)
}

fn ids(v: &Value) -> impl Iterator<Item = usize> + '_ {
    v.as_array().into_iter().flatten().filter_map(|x| x.as_u64().map(|x| x as usize))
}

impl Whisper {
    pub fn load(dir: &Path) -> Result<Self> {
        let config = read_json(&dir.join("config.json"));
        let gen_cfg = read_json(&dir.join("generation_config.json"));
        let pre = read_json(&dir.join("preprocessor_config.json"));

        // No memory arena: Whisper's fixed 30-second window makes its
        // activations big enough that the arena roughly doubles resident
        // memory to save a third of the decode time. A machine that starts
        // swapping loses the whole stream, not just the captions.
        let encoder = create_session(&dir.join("encoder_model.onnx"), false)?;
        let decoder = create_session(&dir.join("decoder_model_merged.onnx"), false)?;

        let input_name = pick_name(&input_names(&encoder), &["input_features"], "encoder input")?;
        let hidden_name =
            pick_name(&output_names(&encoder), &["last_hidden_state", "encoder_hidden_states", "output"], "encoder output")?;

        let n_mels = num(&[(&pre, "feature_size"), (&config, "num_mel_bins")], 80) as usize;
        let frames = pre["nb_max_frames"].as_u64().unwrap_or(config["max_source_positions"].as_u64().unwrap_or(1500) * 2) as usize;
        let samples = pre["n_samples"].as_u64().unwrap_or(480_000) as usize;
        let sample_rate = pre["sampling_rate"].as_u64().unwrap_or(16_000) as usize;
        let features = WhisperFeatures::new(
            sample_rate,
            pre["n_fft"].as_u64().unwrap_or(400) as usize,
            pre["hop_length"].as_u64().unwrap_or(160) as usize,
            n_mels,
        );

        let heads = num(&[(&config, "decoder_num_key_value_heads"), (&config, "decoder_attention_heads")], 6);
        let attn_heads = config["decoder_attention_heads"].as_u64().unwrap_or(heads);
        let hidden = num(&[(&config, "d_model"), (&config, "hidden_size")], 384);
        let layers = match count_layers(&input_names(&decoder)) {
            0 => config["decoder_layers"].as_u64().unwrap_or(4) as usize,
            n => n,
        };
        let greedy = Greedy { layers, kv_heads: heads as usize, head_dim: (hidden as f64 / attn_heads as f64).round() as usize };

        let eos_id = num(&[(&gen_cfg, "eos_token_id"), (&config, "eos_token_id")], 50256) as usize;
        let max_length = num(&[(&gen_cfg, "max_length"), (&config, "max_length")], 448) as usize;

        // decoder_start_token_id, then whatever generation_config pins to
        // which position: <|notimestamps|> at 1 for English-only models; the
        // language and task tokens too for multilingual ones.
        let mut start: Vec<Option<i64>> =
            vec![Some(num(&[(&gen_cfg, "decoder_start_token_id"), (&config, "decoder_start_token_id")], 50257) as i64)];
        let forced = if gen_cfg["forced_decoder_ids"].is_array() { &gen_cfg["forced_decoder_ids"] } else { &config["forced_decoder_ids"] };
        let mut pairs: Vec<(usize, i64)> = forced
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|p| Some((p[0].as_u64()? as usize, p[1].as_i64()?)))
            .collect();
        pairs.sort();
        for (pos, id) in pairs {
            if start.len() <= pos {
                start.resize(pos + 1, None);
            }
            start[pos] = Some(id);
        }
        let start_ids: Vec<i64> = start.into_iter().flatten().collect();

        let suppress_src = if gen_cfg["suppress_tokens"].is_array() { &gen_cfg["suppress_tokens"] } else { &config["suppress_tokens"] };
        let mut suppress: HashSet<usize> = ids(suppress_src).collect();
        // Timestamps are forced off, so the timestamp block is dead
        // vocabulary; emitted anyway it would land in the caption as text.
        if let Some(no_ts) = gen_cfg["no_timestamps_token_id"].as_u64() {
            let vocab = config["vocab_size"].as_u64().unwrap_or(51864);
            suppress.extend((no_ts as usize + 1)..vocab as usize);
        }
        let begin_src =
            if gen_cfg["begin_suppress_tokens"].is_array() { &gen_cfg["begin_suppress_tokens"] } else { &config["begin_suppress_tokens"] };
        let begin_suppress = ids(begin_src).collect();

        Ok(Self {
            detok: tokenizer::load(dir)?,
            encoder,
            decoder,
            features,
            input_name,
            hidden_name,
            n_mels,
            frames,
            samples,
            sample_rate,
            greedy,
            eos_id,
            max_length,
            start_ids,
            suppress,
            begin_suppress,
        })
    }

    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        // Past 30 s is not encodable in one pass; the segmenter never asks.
        let clip = &audio[..audio.len().min(self.samples)];
        let mel = self.features.compute(clip, self.frames);
        let input = f32_tensor(vec![1, self.n_mels as i64, self.frames as i64], mel)?;
        let mut encoded = self.encoder.run(vec![(self.input_name.as_str(), input)]).map_err(err)?;
        let hidden = encoded.remove(&self.hidden_name).ok_or("The encoder returned no hidden states.")?;
        drop(encoded);

        let seconds = clip.len() as f64 / self.sample_rate as f64;
        let budget = (self.max_length - self.start_ids.len()).min((seconds * 8.0).ceil() as usize + 8);
        let ids = self.greedy.generate(
            &mut self.decoder,
            Generate {
                encoder_hidden: &hidden,
                start_ids: &self.start_ids,
                eos_id: self.eos_id,
                max_new_tokens: budget.max(1),
                suppress: Some(&self.suppress),
                begin_suppress: Some(&self.begin_suppress),
            },
        )?;
        Ok(self.detok.decode(&ids).trim().to_string())
    }
}
