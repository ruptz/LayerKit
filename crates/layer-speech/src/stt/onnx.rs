//! ONNX Runtime plumbing shared by the Moonshine, Whisper and Parakeet engines.

use std::collections::HashSet;
use std::path::Path;

use ort::ep::CPU;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{DynValue, Tensor, TensorElementType, ValueType};

pub type Result<T> = std::result::Result<T, String>;

pub fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// How many threads one inference may use. A Layer app shares the machine
/// with a game, Discord and OBS, so taking every core would be the wrong trade
/// even though it would decode fastest. Half the cores, capped at 8; decodes
/// run one at a time, so each gets this budget to itself.
fn thread_budget() -> usize {
    if let Some(n) = crate::env("STT_THREADS").and_then(|v| v.parse().ok()).filter(|&n| n > 0) {
        return n;
    }
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    (cores / 2).clamp(1, 8)
}

/// Whether to keep ONNX Runtime's CPU memory arena: a straight speed-for-memory
/// trade whose worth depends on how large the graph's activations are, so each
/// engine chooses. Measured on a Ryzen 5 5600X, 4-second utterance:
///
///   Whisper Small   arena 3609 MB / 1600 ms      no arena 1779 MB / 2278 ms
///   Whisper Base    arena 1899 MB /  536 ms      no arena 1075 MB /  713 ms
///   Whisper Tiny    arena  833 MB /  298 ms      no arena  382 MB /  423 ms
///
/// Moonshine's arena costs ~100 MB and Parakeet's ~400 MB, so they keep it.
/// <PREFIX>_STT_ARENA=1 or 0 overrides the engine's choice either way.
fn use_arena(preference: bool) -> bool {
    match crate::env("STT_ARENA").as_deref() {
        Some("0") => false,
        Some("1") => true,
        _ => preference,
    }
}

const LIB_FILENAME: &str = if cfg!(windows) {
    "onnxruntime.dll"
} else if cfg!(target_os = "macos") {
    "libonnxruntime.dylib"
} else {
    "libonnxruntime.so"
};

static RUNTIME: std::sync::OnceLock<Result<()>> = std::sync::OnceLock::new();

/// Load the ONNX Runtime once per process: an env override, then the copy
/// bundled with the app, then the vendor/ folder of a source checkout. Only
/// the models that need it ever load it — someone running Vosk never pays.
pub fn init(resource_dir: Option<&Path>) -> Result<()> {
    RUNTIME
        .get_or_init(|| {
            let mut candidates = Vec::new();
            if let Some(p) = crate::env_os("ORT_LIB") {
                candidates.push(std::path::PathBuf::from(p));
            }
            if let Some(r) = resource_dir {
                candidates.push(r.join("onnxruntime").join(LIB_FILENAME));
            }
            // Beside the program itself, as for libvosk.
            if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
                candidates.push(dir.join("onnxruntime").join(LIB_FILENAME));
            }
            // Source checkouts only, as for libvosk.
            if let Some(vendor) = crate::dev_vendor() {
                candidates.push(vendor.join("runtime").join("onnxruntime").join(LIB_FILENAME));
            }
            let lib = candidates.into_iter().find(|p| p.is_file()).ok_or_else(|| {
                format!("Could not find {LIB_FILENAME}, the speech runtime this model needs. Reinstall {}.", crate::app().name)
            })?;
            ort::init_from(&lib)
                .map_err(|e| format!("Could not load the speech runtime ({}): {e}", lib.display()))?
                .with_name(crate::app().env_prefix.to_lowercase())
                .commit();
            Ok(())
        })
        .clone()
}

pub fn create_session(file: &Path, arena: bool) -> Result<Session> {
    let name = file.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    if !file.exists() {
        return Err(format!("Missing model file \"{name}\" in {}.", file.parent().map(|p| p.display().to_string()).unwrap_or_default()));
    }
    let build = || -> ort::Result<Session> {
        Session::builder()?
            .with_execution_providers([CPU::default().with_arena_allocator(use_arena(arena)).build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(false)?
            .with_intra_threads(thread_budget())?
            .with_inter_threads(1)?
            // Idle pool threads otherwise busy-wait for the next run, and
            // Parakeet's joint network is hundreds of tiny runs per clip: the
            // whole pool read as pegged through continuous speech.
            .with_intra_op_spinning(false)?
            .commit_from_file(file)
    };
    build().map_err(|e| format!("Could not load {name}: {e}"))
}

pub fn input_names(s: &Session) -> Vec<String> {
    s.inputs().iter().map(|i| i.name().to_string()).collect()
}

pub fn output_names(s: &Session) -> Vec<String> {
    s.outputs().iter().map(|o| o.name().to_string()).collect()
}

/// Declared element type and shape of a graph input; symbolic axes are -1.
pub fn input_meta(s: &Session, name: &str) -> Option<(TensorElementType, Vec<i64>)> {
    s.inputs().iter().find(|i| i.name() == name).and_then(|i| match i.dtype() {
        ValueType::Tensor { ty, shape, .. } => Some((*ty, shape.iter().copied().collect())),
        _ => None,
    })
}

/// Resolve a graph input/output name from a list of candidates. Exports
/// drift — a re-export renames `input_features` to `mel` — and finding that
/// out at load time beats a confusing failure on the first utterance.
pub fn pick_name(available: &[String], candidates: &[&str], what: &str) -> Result<String> {
    candidates.iter().find(|c| available.iter().any(|a| a == *c)).map(|c| c.to_string()).ok_or_else(|| {
        format!(
            "This model's {what} is not one of the names {} knows ({}). The graph offers: {}.",
            crate::app().name,
            candidates.join(", "),
            available.join(", ")
        )
    })
}

/// Every `past_key_values.N....` group the decoder declares, as a layer count.
pub fn count_layers(inputs: &[String]) -> usize {
    (0..).take_while(|n| inputs.iter().any(|i| *i == format!("past_key_values.{n}.decoder.key"))).count()
}

/// Index of the largest value, ignoring anything in `blocked`.
pub fn argmax(data: &[f32], blocked: Option<&HashSet<usize>>) -> usize {
    let mut best = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in data.iter().enumerate() {
        if blocked.is_some_and(|b| b.contains(&i)) {
            continue;
        }
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

pub fn f32_tensor(shape: Vec<i64>, data: Vec<f32>) -> Result<DynValue> {
    Tensor::from_array((shape, data)).map(|t| t.into_dyn()).map_err(err)
}

pub fn i64_tensor(shape: Vec<i64>, data: Vec<i64>) -> Result<DynValue> {
    Tensor::from_array((shape, data)).map(|t| t.into_dyn()).map_err(err)
}

pub fn i32_tensor(shape: Vec<i64>, data: Vec<i32>) -> Result<DynValue> {
    Tensor::from_array((shape, data)).map(|t| t.into_dyn()).map_err(err)
}

pub fn bool_tensor(value: bool) -> Result<DynValue> {
    Tensor::from_array((vec![1i64], vec![value])).map(|t| t.into_dyn()).map_err(err)
}

/// Shape and data of a float output.
pub fn floats(v: &DynValue) -> Result<(Vec<i64>, &[f32])> {
    let (shape, data) = v.try_extract_tensor::<f32>().map_err(err)?;
    Ok((shape.iter().copied().collect(), data))
}

/// The first element of an integer output, whichever width it was exported as.
pub fn first_int(v: &DynValue) -> Option<i64> {
    if let Ok((_, d)) = v.try_extract_tensor::<i64>() {
        return d.first().copied();
    }
    v.try_extract_tensor::<i32>().ok().and_then(|(_, d)| d.first().map(|&x| x as i64))
}

pub fn read_json(path: &Path) -> serde_json::Value {
    std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(serde_json::json!({}))
}
