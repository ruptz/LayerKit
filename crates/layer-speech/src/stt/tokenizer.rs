//! Detokenisers for the three ONNX engines.
//!
//! Only decoding is needed — the models are only ever asked to transcribe —
//! so this is an id -> piece table plus whatever the `decoder` section of
//! tokenizer.json says to do with the pieces. Two formats:
//!   tokenizer.json  HuggingFace (Whisper: ByteLevel BPE; Moonshine:
//!                   SentencePiece-style BPE with byte fallback)
//!   vocab.txt       "piece id" per line (Parakeet, from NeMo)

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde_json::Value;

/// GPT-2's printable-character -> byte mapping. ByteLevel BPE stores raw
/// bytes as printable code points so the vocabulary is text.
fn byte_decoder() -> HashMap<char, u8> {
    let direct: Vec<u32> = (0x21..=0x7e).chain(0xa1..=0xac).chain(0xae..=0xff).collect();
    let mut map: HashMap<char, u8> = direct.iter().map(|&b| (char::from_u32(b).unwrap(), b as u8)).collect();
    let mut next = 0;
    for b in 0u32..256 {
        if !direct.contains(&b) {
            map.insert(char::from_u32(256 + next).unwrap(), b as u8);
            next += 1;
        }
    }
    map
}

/// A `<0x41>` style byte-fallback piece.
fn byte_fallback(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    (hex.len() == 2).then(|| u8::from_str_radix(hex, 16).ok()).flatten()
}

/// Unknown decoder step types are ignored rather than fatal: a new normaliser
/// upstream should degrade the spacing, not break transcription.
fn flatten(spec: &Value, into: &mut Vec<Value>) {
    if spec["type"] == "Sequence" {
        for d in spec["decoders"].as_array().into_iter().flatten() {
            flatten(d, into);
        }
    } else if spec.is_object() {
        into.push(spec.clone());
    }
}

pub struct Detokenizer {
    pieces: Vec<Option<String>>,
    special: HashSet<usize>,
    byte_level: Option<HashMap<char, u8>>,
    replacements: Vec<(String, String)>,
    /// (content, start, stop)
    strip: Option<(String, usize, usize)>,
}

impl Detokenizer {
    fn new(pieces: Vec<Option<String>>, special: HashSet<usize>, steps: &[Value]) -> Self {
        let byte_level = steps.iter().any(|s| s["type"] == "ByteLevel").then(byte_decoder);
        let mut replacements: Vec<(String, String)> = steps
            .iter()
            .filter(|s| s["type"] == "Replace")
            .filter_map(|s| {
                let p = s["pattern"]["String"].as_str().or(s["pattern"]["Regex"].as_str())?;
                Some((p.to_string(), s["content"].as_str().unwrap_or("").to_string()))
            })
            .collect();
        // Metaspace is the older spelling of Replace("▁", " ").
        if let Some(m) = steps.iter().find(|s| s["type"] == "Metaspace") {
            replacements.push((m["replacement"].as_str().unwrap_or("▁").to_string(), " ".into()));
        }
        let strip = steps.iter().find(|s| s["type"] == "Strip").map(|s| {
            (
                s["content"].as_str().unwrap_or(" ").to_string(),
                s["start"].as_u64().unwrap_or(0) as usize,
                s["stop"].as_u64().unwrap_or(0) as usize,
            )
        });
        Self { pieces, special, byte_level, replacements, strip }
    }

    /// Ids above the last real piece; Parakeet's blank sits at the end.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        // Assembled as bytes: byte-fallback pieces are fragments of one UTF-8
        // sequence and cannot be decoded alone.
        let mut bytes = Vec::new();
        for &id in ids {
            if self.special.contains(&id) {
                continue;
            }
            let Some(Some(piece)) = self.pieces.get(id) else { continue };
            if let Some(b) = byte_fallback(piece) {
                bytes.push(b);
                continue;
            }
            if let Some(table) = &self.byte_level {
                for ch in piece.chars() {
                    match table.get(&ch) {
                        Some(&b) => bytes.push(b),
                        // An added token that was never byte-encoded.
                        None => bytes.extend(ch.to_string().as_bytes()),
                    }
                }
                continue;
            }
            let mut text = piece.clone();
            for (pattern, content) in &self.replacements {
                text = text.replace(pattern.as_str(), content);
            }
            bytes.extend(text.as_bytes());
        }
        let mut text = String::from_utf8_lossy(&bytes).to_string();
        if let Some((ch, start, stop)) = &self.strip {
            for _ in 0..*start {
                match text.strip_prefix(ch.as_str()) {
                    Some(t) => text = t.to_string(),
                    None => break,
                }
            }
            for _ in 0..*stop {
                match text.strip_suffix(ch.as_str()) {
                    Some(t) => text = t.to_string(),
                    None => break,
                }
            }
        }
        text
    }
}

fn put(pieces: &mut Vec<Option<String>>, id: usize, piece: String) {
    if pieces.len() <= id {
        pieces.resize(id + 1, None);
    }
    pieces[id] = Some(piece);
}

fn from_tokenizer_json(file: &Path) -> Result<Detokenizer, String> {
    let json: Value = serde_json::from_slice(&std::fs::read(file).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let mut pieces = Vec::new();
    let mut special = HashSet::new();
    match &json["model"]["vocab"] {
        Value::Object(vocab) => {
            for (piece, id) in vocab {
                if let Some(id) = id.as_u64() {
                    put(&mut pieces, id as usize, piece.clone());
                }
            }
        }
        // Some exports use an array of pairs.
        Value::Array(pairs) => {
            for pair in pairs {
                if let (Some(p), Some(id)) = (pair[0].as_str(), pair[1].as_u64()) {
                    put(&mut pieces, id as usize, p.to_string());
                }
            }
        }
        _ => {}
    }
    for t in json["added_tokens"].as_array().into_iter().flatten() {
        if let (Some(id), Some(content)) = (t["id"].as_u64(), t["content"].as_str()) {
            put(&mut pieces, id as usize, content.to_string());
            if t["special"] == true {
                special.insert(id as usize);
            }
        }
    }
    let mut steps = Vec::new();
    flatten(&json["decoder"], &mut steps);
    Ok(Detokenizer::new(pieces, special, &steps))
}

/// NeMo's SentencePiece vocabulary: one "piece id" per line, `<blk>` last.
fn from_vocab_txt(file: &Path) -> Result<Detokenizer, String> {
    let text = std::fs::read_to_string(file).map_err(|e| e.to_string())?;
    let mut pieces = Vec::new();
    let mut special = HashSet::new();
    for line in text.lines() {
        let Some((piece, id)) = line.rsplit_once(' ') else { continue };
        let Ok(id) = id.parse::<usize>() else { continue };
        if piece.is_empty() {
            continue;
        }
        if piece.len() > 2 && piece.starts_with('<') && piece.ends_with('>') && piece[1..piece.len() - 1].chars().all(|c| c.is_ascii_lowercase()) {
            special.insert(id);
        }
        put(&mut pieces, id, piece.to_string());
    }
    let steps = [
        serde_json::json!({ "type": "Replace", "pattern": { "String": "▁" }, "content": " " }),
        serde_json::json!({ "type": "Strip", "content": " ", "start": 1, "stop": 0 }),
    ];
    Ok(Detokenizer::new(pieces, special, &steps))
}

/// Whichever vocabulary file the model directory carries.
pub fn load(dir: &Path) -> Result<Detokenizer, String> {
    let json = dir.join("tokenizer.json");
    if json.exists() {
        return from_tokenizer_json(&json);
    }
    let txt = dir.join("vocab.txt");
    if txt.exists() {
        return from_vocab_txt(&txt);
    }
    Err(format!(
        "No tokenizer found in \"{}\" (expected tokenizer.json or vocab.txt). \
         Remove the model in the Speech model panel and download it again.",
        dir.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_round_trip() {
        let t = byte_decoder();
        assert_eq!(t[&'Ġ'], b' ');
        assert_eq!(t[&'A'], b'A');
        assert_eq!(t.len(), 256);
    }

    #[test]
    fn fallback_bytes() {
        assert_eq!(byte_fallback("<0x41>"), Some(0x41));
        assert_eq!(byte_fallback("<0x4>"), None);
        assert_eq!(byte_fallback("hello"), None);
    }
}
