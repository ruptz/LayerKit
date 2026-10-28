//! Parakeet TDT 0.6B, on the ONNX Runtime.
//!
//! A transducer, not an encoder-decoder. A TDT ("token and duration
//! transducer") joint network emits a token and how many encoder frames to
//! advance afterwards; predicting the skip is what makes it fast for its size.
//!
//! The package ships NeMo's own feature extractor as a small ONNX graph
//! (nemo128.onnx), used rather than reimplemented: NeMo's front end has
//! per-feature normalisation, pre-emphasis and a log guard that must all match
//! the training recipe, and getting one subtly wrong degrades accuracy in a way
//! that is very hard to notice.

use std::path::Path;

use ort::session::Session;
use ort::value::{DynValue, TensorElementType};

use super::onnx::{
    Result, argmax, create_session, err, f32_tensor, first_int, floats, i32_tensor, i64_tensor, input_meta, input_names,
    output_names, pick_name, read_json,
};
use super::tokenizer::{self, Detokenizer};

/// NeMo's cap on tokens out of a single encoder frame. Without it a
/// duration-0 prediction that keeps emitting would never advance.
const MAX_SYMBOLS_PER_FRAME: usize = 10;

pub struct Parakeet {
    pre: Session,
    encoder: Session,
    joint: Session,
    detok: Detokenizer,
    wav_name: String,
    wav_len_name: String,
    feat_name: String,
    feat_len_name: String,
    enc_in_name: String,
    enc_len_name: String,
    enc_len_i32: bool,
    enc_out_name: String,
    enc_len_out_name: String,
    joint_enc_name: String,
    targets_name: String,
    target_len_name: String,
    targets_i64: bool,
    state_names: Vec<String>,
    state_out_names: Vec<String>,
    state_shapes: Vec<Vec<i64>>,
    joint_out_name: String,
    vocab_size: usize,
    blank_id: usize,
}

fn int_tensor(wide: bool, shape: Vec<i64>, v: i64) -> Result<DynValue> {
    if wide { i64_tensor(shape, vec![v]) } else { i32_tensor(shape, vec![v as i32]) }
}

impl Parakeet {
    pub fn load(dir: &Path) -> Result<Self> {
        let _config = read_json(&dir.join("config.json"));
        // Named for its output width (128 mel bins); accept the other spellings.
        let features_file = ["nemo128.onnx", "nemo80.onnx", "preprocessor.onnx"]
            .iter()
            .map(|f| dir.join(f))
            .find(|f| f.exists())
            .ok_or_else(|| {
                format!(
                    "No feature extractor found in \"{}\" (expected nemo128.onnx). \
                     Remove the model in the Speech model panel and download it again.",
                    dir.display()
                )
            })?;
        let pre = create_session(&features_file, true)?;
        let encoder = create_session(&dir.join("encoder-model.onnx"), true)?;
        let joint = create_session(&dir.join("decoder_joint-model.onnx"), true)?;
        let detok = tokenizer::load(dir)?;

        let (pre_in, pre_out) = (input_names(&pre), output_names(&pre));
        let (enc_in, enc_out) = (input_names(&encoder), output_names(&encoder));
        let (joint_in, joint_out) = (input_names(&joint), output_names(&joint));

        let enc_len_name = pick_name(&enc_in, &["length", "audio_signal_length"], "encoder length input")?;
        let targets_name = pick_name(&joint_in, &["targets"], "joint target input")?;
        let mut state_names: Vec<String> = joint_in.iter().filter(|n| n.starts_with("input_states_")).cloned().collect();
        state_names.sort();
        let mut state_out_names: Vec<String> = joint_out.iter().filter(|n| n.starts_with("output_states_")).cloned().collect();
        state_out_names.sort();
        if state_names.is_empty() {
            return Err("This Parakeet package has no prediction-network state inputs.".into());
        }
        let state_shapes = state_names
            .iter()
            .map(|n| match input_meta(&joint, n) {
                Some((_, shape)) => shape.iter().map(|&d| if d > 0 { d } else { 1 }).collect(),
                None => vec![2, 1, 640],
            })
            .collect();
        // vocab.txt ends with <blk>; everything before it is a real token.
        let vocab_size = detok.len();

        Ok(Self {
            wav_name: pick_name(&pre_in, &["waveforms", "audio_signal", "input"], "features input")?,
            wav_len_name: pick_name(&pre_in, &["waveforms_lens", "length", "input_length"], "features length input")?,
            feat_name: pick_name(&pre_out, &["features", "outputs"], "features output")?,
            feat_len_name: pick_name(&pre_out, &["features_lens", "encoded_lengths", "length"], "features length output")?,
            enc_in_name: pick_name(&enc_in, &["audio_signal", "features"], "encoder input")?,
            enc_len_i32: matches!(input_meta(&encoder, &enc_len_name), Some((TensorElementType::Int32, _))),
            enc_len_name,
            enc_out_name: pick_name(&enc_out, &["outputs", "last_hidden_state"], "encoder output")?,
            enc_len_out_name: pick_name(&enc_out, &["encoded_lengths", "output_lengths"], "encoder length output")?,
            joint_enc_name: pick_name(&joint_in, &["encoder_outputs"], "joint encoder input")?,
            target_len_name: pick_name(&joint_in, &["target_length"], "joint target length input")?,
            targets_i64: matches!(input_meta(&joint, &targets_name), Some((TensorElementType::Int64, _))),
            targets_name,
            joint_out_name: pick_name(&joint_out, &["outputs", "logits"], "joint output")?,
            state_names,
            state_out_names,
            state_shapes,
            vocab_size,
            blank_id: vocab_size - 1,
            pre,
            encoder,
            joint,
            detok,
        })
    }

    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let mut features = self
            .pre
            .run(vec![
                (self.wav_name.as_str(), f32_tensor(vec![1, audio.len() as i64], audio.to_vec())?),
                (self.wav_len_name.as_str(), i64_tensor(vec![1], vec![audio.len() as i64])?),
            ])
            .map_err(err)?;
        let feats = features.remove(&self.feat_name).ok_or("The feature extractor returned nothing.")?;
        let feat_len = features.remove(&self.feat_len_name).ok_or("The feature extractor returned no length.")?;
        drop(features);

        let feat_len = if self.enc_len_i32 { i32_tensor(vec![1], vec![first_int(&feat_len).unwrap_or(0) as i32])? } else { feat_len };
        let encoded = self
            .encoder
            .run(vec![(self.enc_in_name.as_str(), feats), (self.enc_len_name.as_str(), feat_len)])
            .map_err(err)?;
        let (shape, data) = floats(&encoded[self.enc_out_name.as_str()])?;
        // [1, dim, frames] — channel-major, so one frame is a stride apart.
        let (dim, frames) = (shape[1] as usize, shape[2] as usize);
        let total = encoded.get(&self.enc_len_out_name).and_then(first_int).map(|n| n as usize).filter(|&n| n > 0).unwrap_or(frames);
        let enc = data.to_vec();
        drop(encoded);

        let tokens = self.decode(&enc, dim, frames, total.min(frames))?;
        Ok(self.detok.decode(&tokens).trim().to_string())
    }

    fn fresh_states(&self) -> Result<Vec<DynValue>> {
        self.state_shapes
            .iter()
            .map(|dims| f32_tensor(dims.clone(), vec![0.0; dims.iter().product::<i64>() as usize]))
            .collect()
    }

    /// Greedy TDT search. The prediction network's state may only advance when
    /// a real token is emitted — a blank leaves it where it was — so new
    /// states are held aside and only committed on emission.
    fn decode(&mut self, enc: &[f32], dim: usize, frames: usize, usable: usize) -> Result<Vec<usize>> {
        let mut tokens = Vec::new();
        let mut states = self.fresh_states()?;
        // NeMo's label embedding uses the blank as its padding index, so the
        // blank with zeroed states is exactly the start-of-sequence input.
        let mut previous = self.blank_id;
        let mut t = 0;
        let mut emitted_here = 0;
        // Every iteration advances t or emitted_here, both bounded; the cap
        // keeps a malformed graph from hanging the thread anyway.
        let ceiling = usable * MAX_SYMBOLS_PER_FRAME + 64;

        let mut iterations = 0;
        while t < usable && iterations < ceiling {
            iterations += 1;
            let frame: Vec<f32> = (0..dim).map(|c| enc[c * frames + t]).collect();
            let mut feeds: Vec<(String, ort::session::SessionInputValue)> = vec![
                (self.joint_enc_name.clone(), f32_tensor(vec![1, dim as i64, 1], frame)?.into()),
                (self.targets_name.clone(), int_tensor(self.targets_i64, vec![1, 1], previous as i64)?.into()),
                (self.target_len_name.clone(), int_tensor(self.targets_i64, vec![1], 1)?.into()),
            ];
            for (name, s) in self.state_names.iter().zip(&states) {
                feeds.push((name.clone(), s.view().into()));
            }
            let mut out = self.joint.run(feeds).map_err(err)?;
            let (_, logits) = floats(&out[self.joint_out_name.as_str()])?;

            // Vocabulary logits, then one per trained duration: [0, 1, 2, ...].
            let token = argmax(&logits[..self.vocab_size], None);
            let skip = if logits.len() > self.vocab_size { argmax(&logits[self.vocab_size..], None) } else { 0 };

            if token != self.blank_id {
                tokens.push(token);
                previous = token;
                let mut next = Vec::with_capacity(self.state_out_names.len());
                for name in &self.state_out_names {
                    next.push(out.remove(name).ok_or("missing joint state")?);
                }
                drop(out);
                states = next;
                emitted_here += 1;
            }

            if skip > 0 {
                t += skip;
                emitted_here = 0;
            } else if token == self.blank_id || emitted_here >= MAX_SYMBOLS_PER_FRAME {
                // A blank with nowhere to go, or too many tokens off one frame.
                t += 1;
                emitted_here = 0;
            }
        }
        Ok(tokens)
    }
}
