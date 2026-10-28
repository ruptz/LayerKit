//! Local speech-to-text shared by the Layer apps: the engines (`stt`), the
//! microphone gate (`mic`) and installed-model discovery (`models`).
//!
//! Lifted out of ChatterLayer unchanged in behaviour. The few things that were
//! ChatterLayer's own — its name in error messages, its `CHATTERLAYER_*`
//! overrides, the vendor/ folder of its source checkout — now come from the
//! app, through `init`.

pub mod mic;
pub mod models;
pub mod stt;

use std::path::PathBuf;
use std::sync::OnceLock;

/// What the app using this crate is called, and where its source checkout
/// keeps the speech runtimes.
#[derive(Clone, Debug)]
pub struct App {
    /// Shown to the user: "Reinstall ChatterLayer."
    pub name: &'static str,
    /// Prefix of the environment overrides: `CHATTERLAYER` reads
    /// `CHATTERLAYER_STT_THREADS` and so on.
    pub env_prefix: &'static str,
    /// The source checkout's vendor/ folder, searched for the speech runtimes
    /// in debug builds only. Release builds never search a path fixed at build
    /// time: anyone could create that folder on the user's PC.
    pub dev_vendor: Option<PathBuf>,
}

static APP: OnceLock<App> = OnceLock::new();

/// Name the app. Call once at startup, before any model loads; later calls
/// are ignored.
pub fn init(app: App) {
    let _ = APP.set(app);
}

pub(crate) fn app() -> &'static App {
    APP.get_or_init(|| App { name: "the app", env_prefix: "LAYER", dev_vendor: None })
}

/// An environment override, by its name without the app's prefix.
pub(crate) fn env(name: &str) -> Option<String> {
    std::env::var(format!("{}_{name}", app().env_prefix)).ok()
}

pub(crate) fn env_os(name: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(format!("{}_{name}", app().env_prefix))
}

/// The vendor/ folder to search, in debug builds only.
pub(crate) fn dev_vendor() -> Option<PathBuf> {
    if cfg!(debug_assertions) { app().dev_vendor.clone() } else { None }
}
