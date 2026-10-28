//! Installed speech models: finding them, naming them, and ranking them.
//!
//! Model discovery is manifest-driven. Every model an app installs gets a
//! `chatterlayer-model.json` naming its engine, so a directory on disk is
//! self-describing. Vosk models installed before manifests existed are still
//! recognised by their Kaldi layout.
//!
//! The manifest keeps ChatterLayer's name because models already installed
//! carry it; any Layer app can open them.
//!
//! The catalogue itself — what can be downloaded, labels, best-first ranking —
//! belongs to each app (ChatterLayer's `models.json`), and is handed in as a
//! `Catalog`.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{Value, json};

pub const MANIFEST_FILENAME: &str = "chatterlayer-model.json";

/// An app's catalogue and best-first ranking, as data.
pub struct Catalog {
    data: Value,
}

impl Catalog {
    /// From a `models.json`: `{ "catalog": [...], "rank": [...] }`.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        Ok(Self { data: serde_json::from_str(json)? })
    }

    /// No catalogue: every model is named after its folder and none outranks
    /// another.
    pub fn empty() -> Self {
        Self { data: json!({ "catalog": [], "rank": [] }) }
    }

    pub fn entries(&self) -> &[Value] {
        self.data["catalog"].as_array().map(Vec::as_slice).unwrap_or(&[])
    }

    fn rank_of(&self, dir: &str) -> usize {
        let rank = self.data["rank"].as_array().map(Vec::as_slice).unwrap_or(&[]);
        rank.iter().position(|r| r == dir).unwrap_or(rank.len())
    }

    pub fn find_by_dir(&self, dir: &str) -> Option<&Value> {
        self.entries().iter().find(|m| m["dir"] == dir)
    }

    /// Is this directory a usable model, and if so whose?
    pub fn identify(&self, dir: &Path) -> Option<Model> {
        let name = dir.file_name()?.to_string_lossy().to_string();
        let entry = self.find_by_dir(&name);
        let (engine, key, label) = if let Some(m) = read_manifest(dir) {
            (
                m["engine"].as_str()?.to_string(),
                m["key"].as_str().map(str::to_string).or_else(|| entry.and_then(|e| e["key"].as_str().map(str::to_string))),
                entry.map(label_for).or_else(|| m["label"].as_str().map(str::to_string)).unwrap_or_else(|| name.clone()),
            )
        } else if looks_like_vosk(dir) {
            (
                "vosk".to_string(),
                entry.and_then(|e| e["key"].as_str().map(str::to_string)),
                entry.map(label_for).unwrap_or_else(|| name.clone()),
            )
        } else {
            return None;
        };
        Some(Model {
            path: dir.to_path_buf(),
            engine,
            key,
            label,
            max_speakers: entry.and_then(|e| e["maxSpeakers"].as_u64()),
            name,
        })
    }

    /// Every installed model, best-first.
    pub fn list(&self, models_dir: &Path) -> Vec<Model> {
        let Ok(entries) = fs::read_dir(models_dir) else { return Vec::new() };
        let mut models: Vec<Model> = entries
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .filter_map(|e| self.identify(&e.path()))
            .collect();
        models.sort_by(|a, b| self.rank_of(&a.name).cmp(&self.rank_of(&b.name)).then(a.name.cmp(&b.name)));
        models
    }

    /// Catalogue annotated with whether each model is already on disk.
    pub fn with_status(&self, models_dir: &Path) -> Vec<Value> {
        self.entries()
            .iter()
            .map(|m| {
                let target = models_dir.join(m["dir"].as_str().unwrap_or(""));
                let installed = target.exists()
                    && (read_manifest(&target).is_some() || (m["engine"] == "vosk" && looks_like_vosk(&target)));
                let mut out = m.clone();
                out["installed"] = json!(installed);
                out["path"] = json!(target.display().to_string());
                out
            })
            .collect()
    }
}

fn format_size(mb: f64) -> String {
    if mb >= 1000.0 { format!("{:.1} GB", mb / 1000.0) } else { format!("{mb} MB") }
}

/// Picker label: name, approximate size, one short note.
fn label_for(entry: &Value) -> String {
    format!(
        "{} — {}, {}",
        entry["label"].as_str().unwrap_or(""),
        format_size(entry["downloadMB"].as_f64().unwrap_or(0.0)),
        entry["note"].as_str().unwrap_or("")
    )
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub name: String,
    pub path: PathBuf,
    pub engine: String,
    pub key: Option<String>,
    pub label: String,
    pub max_speakers: Option<u64>,
}

fn read_manifest(dir: &Path) -> Option<Value> {
    let m: Value = serde_json::from_slice(&fs::read(dir.join(MANIFEST_FILENAME)).ok()?).ok()?;
    m["engine"].is_string().then_some(m)
}

fn looks_like_vosk(dir: &Path) -> bool {
    dir.join("conf").exists() || dir.join("am").exists()
}
