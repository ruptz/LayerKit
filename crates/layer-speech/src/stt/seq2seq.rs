//! Greedy decoding for the two encoder-decoder engines.
//!
//! Whisper and Moonshine have different front ends, but their ONNX exports
//! present the identical decoder contract — `input_ids`,
//! `encoder_hidden_states`, a `past_key_values.N.{decoder,encoder}.{key,value}`
//! cache and a `use_cache_branch` switch — so the token loop is written once.
//!
//! Greedy rather than beam search: a beam costs N times the compute for an
//! accuracy gain nobody watching live captions would notice.

use std::collections::{HashMap, HashSet};

use ort::session::Session;
use ort::value::DynValue;

use super::onnx::{Result, argmax, bool_tensor, err, f32_tensor, floats, i64_tensor};

/// Both models can fall into a repeating tail — Whisper notoriously so on
/// noise or a clipped word. Stop as soon as the tail is clearly a cycle rather
/// than burn the whole budget on the same phrase eight times.
pub fn looks_looped(tokens: &[usize]) -> bool {
    let n = tokens.len();
    if n < 8 {
        return false;
    }
    (1..=6).take_while(|p| n >= p * 3).any(|p| (0..p * 2).all(|i| tokens[n - 1 - i] == tokens[n - 1 - i - p]))
}

pub struct Greedy {
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

pub struct Generate<'a> {
    pub encoder_hidden: &'a DynValue,
    pub start_ids: &'a [i64],
    pub eos_id: usize,
    pub max_new_tokens: usize,
    /// Never emitted, at any position.
    pub suppress: Option<&'a HashSet<usize>>,
    /// Never emitted as the first token.
    pub begin_suppress: Option<&'a HashSet<usize>>,
}

impl Greedy {
    /// Zero-length cache entries for the first pass, where nothing is cached.
    fn empty_cache(&self) -> Result<Vec<(String, DynValue)>> {
        let mut feeds = Vec::new();
        for i in 0..self.layers {
            for side in ["decoder", "encoder"] {
                for kind in ["key", "value"] {
                    let shape = vec![1, self.kv_heads as i64, 0, self.head_dim as i64];
                    feeds.push((format!("past_key_values.{i}.{side}.{kind}"), f32_tensor(shape, Vec::new())?));
                }
            }
        }
        Ok(feeds)
    }

    /// Generated ids, without the start sequence or EOS.
    pub fn generate(&self, decoder: &mut Session, g: Generate) -> Result<Vec<usize>> {
        let mut input_ids = i64_tensor(vec![1, g.start_ids.len() as i64], g.start_ids.to_vec())?;
        let mut use_cache = false;
        let mut cache: Vec<(String, DynValue)> = self.empty_cache()?;
        // The cross-attention cache depends on the encoder output alone, so it
        // is taken from the first pass and carried forward unchanged.
        let mut encoder_cache: HashMap<String, DynValue> = HashMap::new();

        let first_blocked: Option<HashSet<usize>> = g.begin_suppress.map(|b| {
            let mut all = b.clone();
            all.extend(g.suppress.into_iter().flatten());
            all
        });

        let mut generated = Vec::new();
        for step in 0..g.max_new_tokens {
            let mut feeds: Vec<(String, ort::session::SessionInputValue)> = vec![
                ("input_ids".into(), input_ids.into()),
                ("encoder_hidden_states".into(), g.encoder_hidden.view().into()),
                ("use_cache_branch".into(), bool_tensor(use_cache)?.into()),
            ];
            for (name, v) in cache.drain(..) {
                feeds.push((name, v.into()));
            }
            // Re-fed every step without being recomputed.
            for (name, v) in &encoder_cache {
                feeds.push((name.clone(), v.view().into()));
            }
            let mut out = decoder.run(feeds).map_err(err)?;

            let (shape, logits) = floats(&out["logits"])?;
            let vocab = *shape.last().unwrap_or(&0) as usize;
            if vocab == 0 {
                return Err("The decoder returned no logits.".into());
            }
            // Only the last position matters: on the priming pass the decoder
            // scores every start token, and all but the last are known.
            let last = &logits[logits.len() - vocab..];
            let blocked = if step == 0 && first_blocked.is_some() { first_blocked.as_ref() } else { g.suppress };
            let token = argmax(last, blocked);

            if token == g.eos_id {
                break;
            }
            generated.push(token);
            if looks_looped(&generated) {
                generated.truncate(generated.len().saturating_sub(2));
                break;
            }

            let names: Vec<String> = out.keys().filter(|k| k.starts_with("present")).map(str::to_string).collect();
            for name in names {
                let as_past = name.replacen("present", "past_key_values", 1);
                let value = out.remove(&name).ok_or("missing cache output")?;
                if name.contains(".encoder.") {
                        encoder_cache.entry(as_past).or_insert(value);
                } else {
                    cache.push((as_past, value));
                }
            }
            drop(out);
            input_ids = i64_tensor(vec![1, 1], vec![token as i64])?;
            use_cache = true;
        }
        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::looks_looped;

    #[test]
    fn loops() {
        assert!(looks_looped(&[1, 2, 3, 4, 5, 5, 5, 5, 5]));
        assert!(looks_looped(&[9, 1, 2, 1, 2, 1, 2, 1, 2]));
        assert!(!looks_looped(&[1, 2, 3, 4, 5, 6, 7, 8, 9]));
    }
}
