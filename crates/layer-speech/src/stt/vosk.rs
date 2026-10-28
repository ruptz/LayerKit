//! The official libvosk, opened at runtime like the Electron build's koffi
//! binding did, so the app builds without it and finds whichever copy ships
//! beside it. Signatures come from vendor/.../vosk_api.h.

use std::ffi::{CStr, CString, c_char, c_float, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libloading::Library;

const LIB_FILENAME: &str = if cfg!(windows) {
    "libvosk.dll"
} else if cfg!(target_os = "macos") {
    "libvosk.dylib"
} else {
    "libvosk.so"
};

const SAMPLE_RATE: c_float = 16000.0;

type Ptr = *mut c_void;

struct Api {
    _lib: Library,
    model_free: unsafe extern "C" fn(Ptr),
    recognizer_new: unsafe extern "C" fn(Ptr, c_float) -> Ptr,
    accept_waveform: unsafe extern "C" fn(Ptr, *const c_char, c_int) -> c_int,
    result: unsafe extern "C" fn(Ptr) -> *const c_char,
    partial_result: unsafe extern "C" fn(Ptr) -> *const c_char,
    final_result: unsafe extern "C" fn(Ptr) -> *const c_char,
    recognizer_free: unsafe extern "C" fn(Ptr),
}

/// libvosk: an env override, then the copy bundled with the app, then the
/// vendor/ folder of a source checkout (the upstream zips extract into a
/// versioned folder such as vendor/vosk-win64-0.3.45/).
pub fn find_library(resource_dir: Option<&Path>) -> Result<PathBuf, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = crate::env_os("VOSK_LIB") {
        let p = PathBuf::from(p);
        candidates.push(if p.is_dir() { p.join(LIB_FILENAME) } else { p });
    }
    if let Some(r) = resource_dir {
        candidates.push(r.join("vosk").join(LIB_FILENAME));
    }
    // Beside the program itself: how the WidgetCake zip lays it out on Linux,
    // where Tauri's own resource folder is `../lib/<app>` or the AppImage.
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        candidates.push(dir.join("vosk").join(LIB_FILENAME));
    }
    // Source checkouts only. The path is fixed at build time, so a release
    // would otherwise search a folder on the user's PC anyone could create.
    if let Some(vendor) = crate::dev_vendor() {
        // Where the app's fetch-runtimes script puts it.
        candidates.push(vendor.join("runtime").join("vosk").join(LIB_FILENAME));
    }
    candidates
        .into_iter()
        .find(|p| p.is_file())
        .ok_or_else(|| format!("Could not find {LIB_FILENAME}, the Vosk runtime. Reinstall {}.", crate::app().name))
}

fn open(path: &Path) -> Result<Library, libloading::Error> {
    // Its own folder first when resolving libvosk's dependencies (the MinGW
    // runtime DLLs shipped beside it), not the app's.
    #[cfg(windows)]
    unsafe {
        use libloading::os::windows::{LOAD_WITH_ALTERED_SEARCH_PATH, Library as WinLibrary};
        WinLibrary::load_with_flags(path, LOAD_WITH_ALTERED_SEARCH_PATH).map(Into::into)
    }
    #[cfg(not(windows))]
    unsafe {
        Library::new(path)
    }
}

impl Api {
    fn load(path: &Path) -> Result<Self, String> {
        let fail = |e: libloading::Error| format!("Could not load the Vosk runtime ({}): {e}", path.display());
        // SAFETY: the symbols are declared with the signatures in vosk_api.h,
        // and the function pointers never outlive `_lib`, which is stored
        // alongside them.
        unsafe {
            let lib = open(path).map_err(fail)?;
            let set_log_level: unsafe extern "C" fn(c_int) = *lib.get(b"vosk_set_log_level\0").map_err(fail)?;
            // Kaldi is extremely chatty on stderr; -1 silences everything.
            set_log_level(-1);
            Ok(Self {
                model_free: *lib.get(b"vosk_model_free\0").map_err(fail)?,
                recognizer_new: *lib.get(b"vosk_recognizer_new\0").map_err(fail)?,
                accept_waveform: *lib.get(b"vosk_recognizer_accept_waveform\0").map_err(fail)?,
                result: *lib.get(b"vosk_recognizer_result\0").map_err(fail)?,
                partial_result: *lib.get(b"vosk_recognizer_partial_result\0").map_err(fail)?,
                final_result: *lib.get(b"vosk_recognizer_final_result\0").map_err(fail)?,
                recognizer_free: *lib.get(b"vosk_recognizer_free\0").map_err(fail)?,
                _lib: lib,
            })
        }
    }
}

/// The acoustic model. This is the expensive object — 116 MB for the small
/// model, 6.8 GB for Gigaspeech — and is shared by every recogniser, so adding
/// a speaker costs a recogniser (~12 MB) rather than another copy.
pub struct VoskModel {
    api: Arc<Api>,
    handle: Ptr,
    pub library_path: PathBuf,
}

// SAFETY: a Vosk model is immutable once loaded and documented as shareable
// between recognisers on any thread.
unsafe impl Send for VoskModel {}
unsafe impl Sync for VoskModel {}

impl VoskModel {
    pub fn load(dir: &Path, library_path: PathBuf) -> Result<Self, String> {
        if !dir.join("conf").exists() && !dir.join("am").exists() {
            return Err(format!(
                "\"{}\" does not look like a Vosk model (no conf/ or am/ inside).",
                dir.display()
            ));
        }
        let api = Arc::new(Api::load(&library_path)?);
        let c_path = CString::new(dir.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        // SAFETY: symbol from vosk_api.h; c_path outlives the call.
        let handle = unsafe {
            let model_new: libloading::Symbol<unsafe extern "C" fn(*const c_char) -> Ptr> =
                api._lib.get(b"vosk_model_new\0").map_err(|e| e.to_string())?;
            model_new(c_path.as_ptr())
        };
        if handle.is_null() {
            return Err(format!("Vosk failed to load the model at \"{}\".", dir.display()));
        }
        Ok(Self { api, handle, library_path })
    }

    pub fn session(&self) -> Result<VoskSession, String> {
        // SAFETY: the model handle is live for as long as `self`.
        let handle = unsafe { (self.api.recognizer_new)(self.handle, SAMPLE_RATE) };
        if handle.is_null() {
            return Err("Failed to create a Vosk recogniser.".into());
        }
        Ok(VoskSession { api: self.api.clone(), handle })
    }
}

impl Drop for VoskModel {
    fn drop(&mut self) {
        // SAFETY: handle came from vosk_model_new; recognisers hold their own
        // reference inside libvosk, so freeing here is safe even if one lives on.
        unsafe { (self.api.model_free)(self.handle) }
    }
}

/// One streaming decoder, created per selected speaker and freed on toggle-off.
pub struct VoskSession {
    api: Arc<Api>,
    handle: Ptr,
}

unsafe impl Send for VoskSession {}

fn text_of(json: *const c_char, field: &str) -> String {
    if json.is_null() {
        return String::new();
    }
    // SAFETY: libvosk returns a NUL-terminated string valid until the next
    // call on the same recogniser; it is copied out at once.
    let raw = unsafe { CStr::from_ptr(json) }.to_string_lossy();
    serde_json::from_str::<serde_json::Value>(&raw)
        .ok()
        .and_then(|v| v[field].as_str().map(str::to_string))
        .unwrap_or_default()
}

impl VoskSession {
    /// Feed 16 kHz mono PCM. True when an utterance just ended.
    pub fn accept(&mut self, pcm: &[i16]) -> bool {
        // `len` is a byte count, not a sample count.
        let len = (pcm.len() * 2) as c_int;
        // SAFETY: pcm is valid for `len` bytes for the duration of the call.
        unsafe { (self.api.accept_waveform)(self.handle, pcm.as_ptr().cast(), len) == 1 }
    }

    /// Text of the utterance that `accept` just reported.
    pub fn result(&mut self) -> String {
        text_of(unsafe { (self.api.result)(self.handle) }, "text")
    }

    /// Current hypothesis. Cheap; safe to poll.
    pub fn partial(&mut self) -> String {
        text_of(unsafe { (self.api.partial_result)(self.handle) }, "partial")
    }

    /// Close the utterance early and return whatever was buffered.
    pub fn finalize(&mut self) -> String {
        text_of(unsafe { (self.api.final_result)(self.handle) }, "text")
    }
}

impl Drop for VoskSession {
    fn drop(&mut self) {
        // SAFETY: handle came from vosk_recognizer_new and is freed once.
        unsafe { (self.api.recognizer_free)(self.handle) }
    }
}
