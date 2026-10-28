//! Moonshine, on the ONNX Runtime.
//!
//! Its encoder takes raw 16 kHz samples and downsamples with convolutions, so
//! there is no mel spectrogram here. It has no fixed input window either: cost
//! is proportional to the utterance rather than padded to 30 seconds, which
//! makes it the best of the three for frequent partials.

use std::path::Path;

use ort::session::Session;

use super::onnx::{Result, count_layers, create_session, err, f32_tensor, input_names, output_names, pick_name, read_json};
use super::seq2seq::{Generate, Greedy};
use super::tokenizer::{self, Detokenizer};

/// The convolution stack strides by 384 in total, so a very short clip
/// reduces to zero frames and the graph fails. Half a second is well clear.
const MIN_SAMPLES: usize = 8000;

/// Upstream's guidance: about six tokens per second of speech.
const TOKENS_PER_SECOND: f64 = 6.0;

pub struct Moonshine {
    encoder: Session,
    decoder: Session,
    detok: Detokenizer,
    input_name: String,
    hidden_name: String,
    sample_rate: usize,
    greedy: Greedy,
    start_ids: Vec<i64>,
    eos_id: usize,
    max_tokens: usize,
}

impl Moonshine {
    pub fn load(dir: &Path) -> Result<Self> {
        let config = read_json(&dir.join("config.json"));
        let gen_cfg = read_json(&dir.join("generation_config.json"));
        let pre = read_json(&dir.join("preprocessor_config.json"));
        let encoder = create_session(&dir.join("encoder_model.onnx"), true)?;
        let decoder = create_session(&dir.join("decoder_model_merged.onnx"), true)?;

        let input_name = pick_name(&input_names(&encoder), &["input_values", "input_features", "audio"], "encoder input")?;
        let hidden_name =
            pick_name(&output_names(&encoder), &["last_hidden_state", "encoder_hidden_states", "output"], "encoder output")?;

        let heads = config["decoder_num_key_value_heads"].as_u64().or(config["num_key_value_heads"].as_u64()).unwrap_or(8);
        let attn_heads = config["decoder_num_attention_heads"].as_u64().or(config["num_attention_heads"].as_u64()).unwrap_or(heads);
        let hidden = config["hidden_size"].as_u64().unwrap_or(416);
        let layers = match count_layers(&input_names(&decoder)) {
            0 => config["decoder_num_hidden_layers"].as_u64().unwrap_or(8) as usize,
            n => n,
        };

        Ok(Self {
            detok: tokenizer::load(dir)?,
            encoder,
            decoder,
            input_name,
            hidden_name,
            sample_rate: pre["sampling_rate"].as_u64().unwrap_or(16_000) as usize,
            greedy: Greedy { layers, kv_heads: heads as usize, head_dim: (hidden as f64 / attn_heads as f64).round() as usize },
            start_ids: vec![gen_cfg["decoder_start_token_id"].as_i64().or(config["decoder_start_token_id"].as_i64()).unwrap_or(1)],
            eos_id: gen_cfg["eos_token_id"].as_u64().or(config["eos_token_id"].as_u64()).unwrap_or(2) as usize,
            max_tokens: config["max_position_embeddings"].as_u64().unwrap_or(512).min(256) as usize,
        })
    }

    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let mut clip = audio.to_vec();
        if clip.len() < MIN_SAMPLES {
            clip.resize(MIN_SAMPLES, 0.0);
        }
        let seconds = clip.len() as f64 / self.sample_rate as f64;
        let input = f32_tensor(vec![1, clip.len() as i64], clip)?;
        let mut encoded = self.encoder.run(vec![(self.input_name.as_str(), input)]).map_err(err)?;
        let hidden = encoded.remove(&self.hidden_name).ok_or("The encoder returned no hidden states.")?;
        drop(encoded);

        let budget = self.max_tokens.min((seconds * TOKENS_PER_SECOND).ceil() as usize + 8);
        let ids = self.greedy.generate(
            &mut self.decoder,
            Generate {
                encoder_hidden: &hidden,
                start_ids: &self.start_ids,
                eos_id: self.eos_id,
                max_new_tokens: budget.max(1),
                suppress: None,
                begin_suppress: None,
            },
        )?;
        Ok(self.detok.decode(&ids).trim().to_string())
    }
}
