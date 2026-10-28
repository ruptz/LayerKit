//! The speech worker. One per connection, whichever engine is selected.
//!
//! Decoding is CPU-bound, so it runs on its own thread, apart from the async
//! runtime that carries the Discord connection. The protocol is
//! engine-independent: `Add`/`Remove`/`Audio`/`Flush` in, `Partial`/`Final`/
//! `PartialCleared` out. Nothing upstream knows which engine is loaded.

mod features;
mod moonshine;
mod onnx;
mod parakeet;
mod segmenter;
mod seq2seq;
mod tokenizer;
mod vosk;
mod whisper;

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use crate::models::Model;
use segmenter::{Segmenter, Verdict};

/// Streaming engines give partials away free, so the only limit is UI churn.
const STREAM_PARTIAL: Duration = Duration::from_millis(120);

/// Floor on how often a clip engine may be asked for a partial. The real
/// interval comes from measured decode time — see `partial_interval`.
const BASE_PARTIAL: Duration = Duration::from_millis(700);

/// A speaker waits this many decode-lengths between partials, so live captions
/// keep the decoder busy at most a quarter of the time. At 2 they held it half
/// the time, and on Parakeet that read as ~17% of a 12-thread machine through
/// any long turn; at 4 it is ~10%, with a caption about every 1.5 s.
const PARTIAL_WEIGHT: u32 = 4;

/// Discord sends nothing while a speaker is quiet, so silence can't be heard —
/// there's no audio. The speaking-end flush covers the normal case; this covers
/// the times it never comes.
const IDLE_FINALIZE: Duration = Duration::from_millis(450);
const IDLE_TICK: Duration = Duration::from_millis(150);

/// Longest clip handed to an engine. Whisper physically can't encode more than
/// its 30-second window, and for the others a long clip is a long wait.
const MAX_UTTERANCE_SECONDS: usize = 20;

/// Live captions decode only the end of a long utterance; anything older is
/// decoded once, frozen as text, and the window moves on. Re-decoding the whole
/// utterance on every update made a long turn cost ten times its final
/// transcript. The final still decodes the whole clip, so context lost at a
/// freeze point only ever shows in the live caption.
const PARTIAL_WINDOW: usize = 16_000 * 8;
/// A freeze leaves at least this much behind it, so the window always opens
/// with some context...
const PARTIAL_KEEP: usize = 16_000 * 3;
/// ...and freezes at least this much, so it never decodes slivers.
const PARTIAL_MIN_FREEZE: usize = 16_000 * 2;

/// A speaker whose audio fails on essentially every packet would otherwise
/// report one error per 20 ms. Report the first, stay quiet through a short run
/// (~240 ms of packets), then stop feeding that speaker.
const MAX_PUSH_FAILURES: u32 = 12;

pub enum In {
    Add(String),
    Remove(String),
    /// 16 kHz mono PCM.
    Audio(String, Vec<i16>),
    Flush(String),
    Shutdown,
}

#[derive(Debug)]
pub enum Out {
    Ready { engine: String, engine_label: String, streaming: bool, library_path: Option<String>, load_ms: u128 },
    /// The model never loaded, or the worker died. Nothing else follows.
    Fatal(String),
    Partial { user_id: String, text: String },
    Final { user_id: String, text: String },
    PartialCleared { user_id: String },
    Error { user_id: Option<String>, message: String, fatal: bool },
    SpeakersChanged,
}

pub struct Speech {
    tx: mpsc::Sender<In>,
}

impl Speech {
    /// Start a worker for `model`. Its first message is `Ready` or `Fatal`.
    pub fn spawn(model: &Model, resource_dir: Option<&Path>, out: impl Fn(Out) + Send + 'static) -> Self {
        let (tx, rx) = mpsc::channel();
        let model = model.clone();
        let library = vosk::find_library(resource_dir);
        let runtime_dir = resource_dir.map(Path::to_path_buf);
        thread::Builder::new()
            .name("speech".into())
            .spawn(move || {
                // A panic in here is ours, not the model's, and must cost the
                // worker — which the engine restarts — not the app.
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(model, library, runtime_dir, rx, &out)));
                if outcome.is_err() {
                    out(Out::Fatal("The speech engine crashed.".into()));
                }
            })
            .expect("the speech thread");
        Self { tx }
    }

    /// For the voice receiver, which feeds audio without going through the
    /// engine's state.
    pub fn sender(&self) -> mpsc::Sender<In> {
        self.tx.clone()
    }
}

impl Drop for Speech {
    fn drop(&mut self) {
        let _ = self.tx.send(In::Shutdown);
    }
}

fn label_of(engine: &str) -> &'static str {
    match engine {
        "vosk" => "Vosk",
        "whisper" => "Whisper",
        "moonshine" => "Moonshine",
        "parakeet" => "Parakeet",
        _ => "Unknown",
    }
}

/// Whisper, Moonshine and Parakeet: buffer an utterance, then transcribe it.
enum ClipEngine {
    Whisper(whisper::Whisper),
    Moonshine(moonshine::Moonshine),
    Parakeet(parakeet::Parakeet),
}

impl ClipEngine {
    fn transcribe(&mut self, audio: &[f32]) -> Result<String, String> {
        match self {
            Self::Whisper(e) => e.transcribe(audio),
            Self::Moonshine(e) => e.transcribe(audio),
            Self::Parakeet(e) => e.transcribe(audio),
        }
    }

    /// Whisper's encoder runs over a padded 30-second window whatever the clip
    /// length, so a shorter clip saves it nothing and a freeze costs it a whole
    /// extra decode.
    fn windowed(&self) -> bool {
        !matches!(self, Self::Whisper(_))
    }
}

fn run(model: Model, library: Result<PathBuf, String>, runtime_dir: Option<PathBuf>, rx: mpsc::Receiver<In>, out: &dyn Fn(Out)) {
    let runtime = || onnx::init(runtime_dir.as_deref());
    let t0 = Instant::now();
    let ready = |library_path: Option<String>| Out::Ready {
        engine: model.engine.clone(),
        engine_label: label_of(&model.engine).into(),
        streaming: model.engine == "vosk",
        library_path,
        load_ms: t0.elapsed().as_millis(),
    };
    let loaded = match model.engine.as_str() {
        "vosk" => library.and_then(|lib| vosk::VoskModel::load(&model.path, lib)).map(|m| {
            out(ready(Some(m.library_path.display().to_string())));
            run_streaming(m, &rx, out);
            None
        }),
        "whisper" => runtime().and_then(|_| whisper::Whisper::load(&model.path)).map(|e| Some(ClipEngine::Whisper(e))),
        "moonshine" => runtime().and_then(|_| moonshine::Moonshine::load(&model.path)).map(|e| Some(ClipEngine::Moonshine(e))),
        "parakeet" => runtime().and_then(|_| parakeet::Parakeet::load(&model.path)).map(|e| Some(ClipEngine::Parakeet(e))),
        other => Err(format!("Unknown speech engine \"{other}\".")),
    };
    match loaded {
        Err(message) => out(Out::Fatal(message)),
        // Vosk ran and has shut down.
        Ok(None) => {}
        Ok(Some(clip)) => {
            out(ready(None));
            Clips::new(clip, out).run(&rx);
        }
    }
}

// ------------------------------------------------------------- streaming ---

/// Vosk: feed it audio, it says where the utterances are. Returns on shutdown.
fn run_streaming(engine: vosk::VoskModel, rx: &mpsc::Receiver<In>, out: &dyn Fn(Out)) {
    struct Speaker {
        session: vosk::VoskSession,
        last_partial: String,
        last_partial_at: Option<Instant>,
    }
    let mut speakers: HashMap<String, Speaker> = HashMap::new();
    let mut failures: HashMap<String, u32> = HashMap::new();

    let flush = |user_id: &str, s: &mut Speaker| {
        let text = s.session.finalize();
        s.last_partial.clear();
        out(if text.is_empty() {
            Out::PartialCleared { user_id: user_id.into() }
        } else {
            Out::Final { user_id: user_id.into(), text }
        });
    };

    loop {
        let Ok(msg) = rx.recv() else { return };
        match msg {
            In::Add(user_id) => {
                if speakers.contains_key(&user_id) {
                    continue;
                }
                match engine.session() {
                    Ok(session) => {
                        speakers.insert(user_id, Speaker { session, last_partial: String::new(), last_partial_at: None });
                        out(Out::SpeakersChanged);
                    }
                    Err(e) => out(Out::Error { user_id: Some(user_id), message: format!("recognizer init failed: {e}"), fatal: false }),
                }
            }
            In::Remove(user_id) => {
                failures.remove(&user_id);
                // Anything still buffered goes out, so the last words aren't
                // lost on toggle-off.
                if let Some(mut s) = speakers.remove(&user_id) {
                    flush(&user_id, &mut s);
                    out(Out::SpeakersChanged);
                }
            }
            In::Audio(user_id, pcm) => {
                let Some(s) = speakers.get_mut(&user_id) else { continue };
                let step = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if s.session.accept(&pcm) {
                        s.last_partial.clear();
                        let text = s.session.result();
                        return (!text.is_empty()).then(|| Out::Final { user_id: user_id.clone(), text });
                    }
                    let now = Instant::now();
                    if s.last_partial_at.is_some_and(|t| now - t < STREAM_PARTIAL) {
                        return None;
                    }
                    s.last_partial_at = Some(now);
                    let partial = s.session.partial();
                    if partial.is_empty() || partial == s.last_partial {
                        return None;
                    }
                    s.last_partial = partial.clone();
                    Some(Out::Partial { user_id: user_id.clone(), text: partial })
                }));
                match step {
                    Ok(msg) => {
                        failures.remove(&user_id);
                        if let Some(msg) = msg {
                            out(msg);
                        }
                    }
                    Err(_) => push_failed(&user_id, &mut failures, out, || {
                        speakers.remove(&user_id);
                    }),
                }
            }
            In::Flush(user_id) => {
                if let Some(s) = speakers.get_mut(&user_id) {
                    flush(&user_id, s);
                }
            }
            In::Shutdown => return,
        }
    }
}

fn push_failed(user_id: &str, failures: &mut HashMap<String, u32>, out: &dyn Fn(Out), drop_speaker: impl FnOnce()) {
    let n = failures.entry(user_id.to_string()).or_default();
    *n += 1;
    if *n == 1 {
        out(Out::Error { user_id: Some(user_id.into()), message: "decode failed".into(), fatal: false });
    } else if *n >= MAX_PUSH_FAILURES {
        failures.remove(user_id);
        drop_speaker();
        out(Out::Error {
            user_id: Some(user_id.into()),
            message: "speech recognition kept failing and has been stopped for this speaker — toggle them off and back on to retry".into(),
            fatal: true,
        });
        out(Out::PartialCleared { user_id: user_id.into() });
        out(Out::SpeakersChanged);
    }
}

// ------------------------------------------------------------- segmented ---

struct ClipSpeaker {
    seg: Segmenter,
    last_audio_at: Instant,
    last_partial: String,
    last_partial_at: Option<Instant>,
    /// Bumped on every close, so a late partial from the last one is dropped.
    epoch: u64,
    /// How far into the utterance the live caption is frozen, and its text.
    frozen_at: Option<usize>,
    frozen: String,
}

impl ClipSpeaker {
    fn close(&mut self) {
        self.epoch += 1;
        self.frozen_at = None;
        self.frozen.clear();
    }
}

/// Finals are guaranteed — queued, and always run. Partials are speculative:
/// at most one waiting per speaker, skipped whenever a final is waiting, and
/// discarded if their utterance has already closed. That asymmetry is how
/// several speakers share one CPU without any of them losing a caption.
struct Clips<'a> {
    engine: ClipEngine,
    out: &'a dyn Fn(Out),
    speakers: HashMap<String, ClipSpeaker>,
    finals: VecDeque<(String, Vec<f32>)>,
    /// user -> epoch, in arrival order. The audio is taken when the decode
    /// runs, so it's as fresh as it can be.
    partials: VecDeque<(String, u64)>,
    last_run: Duration,
}

impl<'a> Clips<'a> {
    fn new(engine: ClipEngine, out: &'a dyn Fn(Out)) -> Self {
        Self { engine, out, speakers: HashMap::new(), finals: VecDeque::new(), partials: VecDeque::new(), last_run: Duration::ZERO }
    }

    fn run(mut self, rx: &mpsc::Receiver<In>) {
        loop {
            // Decodes block this thread, so audio piles up meanwhile; take all
            // of it before deciding what to decode next.
            let wait = if self.finals.is_empty() && self.partials.is_empty() { IDLE_TICK } else { Duration::ZERO };
            match rx.recv_timeout(wait) {
                Ok(In::Shutdown) | Err(RecvTimeoutError::Disconnected) => return,
                Ok(msg) => {
                    if !self.handle(msg) {
                        return;
                    }
                    while let Ok(msg) = rx.try_recv() {
                        if !self.handle(msg) {
                            return;
                        }
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
            }
            self.idle_tick();
            self.decode_one();
        }
    }

    /// False on shutdown.
    fn handle(&mut self, msg: In) -> bool {
        match msg {
            In::Add(user_id) => {
                if !self.speakers.contains_key(&user_id) {
                    self.speakers.insert(
                        user_id,
                        ClipSpeaker {
                            seg: Segmenter::new(MAX_UTTERANCE_SECONDS),
                            last_audio_at: Instant::now(),
                            last_partial: String::new(),
                            last_partial_at: None,
                            epoch: 0,
                            frozen_at: None,
                            frozen: String::new(),
                        },
                    );
                    (self.out)(Out::SpeakersChanged);
                }
            }
            In::Remove(user_id) => {
                // The last words still go out on toggle-off.
                self.flush(&user_id);
                self.cancel_partials(&user_id);
                if self.speakers.remove(&user_id).is_some() {
                    (self.out)(Out::SpeakersChanged);
                }
            }
            In::Audio(user_id, pcm) => {
                let Some(s) = self.speakers.get_mut(&user_id) else { return true };
                let samples: Vec<f32> = pcm.iter().map(|&x| x as f32 / 32768.0).collect();
                s.seg.push(&samples);
                s.last_audio_at = Instant::now();
                match s.seg.poll() {
                    Some(Verdict::Final) => self.finalize(&user_id),
                    Some(Verdict::Discard) => self.discard(&user_id),
                    None => self.maybe_partial(&user_id),
                }
            }
            In::Flush(user_id) => self.flush(&user_id),
            In::Shutdown => return false,
        }
        true
    }

    fn flush(&mut self, user_id: &str) {
        if self.speakers.get(user_id).is_some_and(|s| !s.seg.is_empty()) {
            self.finalize(user_id);
        }
    }

    fn cancel_partials(&mut self, user_id: &str) {
        self.partials.retain(|(u, _)| u != user_id);
    }

    fn clear_partial(&mut self, user_id: &str) {
        if let Some(s) = self.speakers.get_mut(user_id) {
            if !s.last_partial.is_empty() {
                s.last_partial.clear();
                (self.out)(Out::PartialCleared { user_id: user_id.into() });
            }
        }
    }

    fn finalize(&mut self, user_id: &str) {
        let Some(s) = self.speakers.get_mut(user_id) else { return };
        let clip = s.seg.take();
        s.close();
        self.cancel_partials(user_id);
        match clip {
            Some(clip) => self.finals.push_back((user_id.to_string(), clip)),
            None => self.clear_partial(user_id),
        }
    }

    /// Nothing in the buffer was speech. Don't spend a decode on it.
    fn discard(&mut self, user_id: &str) {
        let Some(s) = self.speakers.get_mut(user_id) else { return };
        s.seg.reset();
        s.close();
        self.cancel_partials(user_id);
        self.clear_partial(user_id);
    }

    /// How long a speaker waits between speculative decodes. Derived from
    /// what decodes cost on this machine, because the same model is minutes
    /// apart on a laptop and a desktop: one speaker on a fast model sits at the
    /// floor and feels live; seven on Parakeet back off to several seconds and
    /// the CPU goes to finals instead.
    fn partial_interval(&self) -> Duration {
        let active = self.speakers.len().max(1) as u32;
        BASE_PARTIAL.max(self.last_run * PARTIAL_WEIGHT * active)
    }

    fn maybe_partial(&mut self, user_id: &str) {
        if crate::env("STT_PARTIALS").as_deref() == Some("0") || !self.finals.is_empty() {
            return;
        }
        let interval = self.partial_interval();
        let Some(s) = self.speakers.get_mut(user_id) else { return };
        let now = Instant::now();
        if s.last_partial_at.is_some_and(|t| now - t < interval) {
            return;
        }
        if !s.seg.has_speech() {
            return;
        }
        s.last_partial_at = Some(now);
        let epoch = s.epoch;
        // A newer partial replaces the waiting one.
        self.cancel_partials(user_id);
        self.partials.push_back((user_id.to_string(), epoch));
    }

    /// Close utterances for speakers who have simply gone quiet.
    fn idle_tick(&mut self) {
        let now = Instant::now();
        let quiet: Vec<(String, bool)> = self
            .speakers
            .iter()
            .filter(|(_, s)| !s.seg.is_empty() && now - s.last_audio_at >= IDLE_FINALIZE)
            .map(|(u, s)| (u.clone(), s.seg.has_speech()))
            .collect();
        for (user_id, speech) in quiet {
            if speech { self.finalize(&user_id) } else { self.discard(&user_id) }
        }
    }

    fn timed(&mut self, clip: &[f32]) -> Result<String, String> {
        let started = Instant::now();
        let result = self.engine.transcribe(clip);
        self.last_run = started.elapsed();
        result
    }

    fn decode_one(&mut self) {
        if let Some((user_id, clip)) = self.finals.pop_front() {
            let result = self.timed(&clip);
            if let Some(s) = self.speakers.get_mut(&user_id) {
                s.last_partial.clear();
            }
            match result {
                Ok(text) if !text.trim().is_empty() => (self.out)(Out::Final { user_id, text: text.trim().into() }),
                Ok(_) => (self.out)(Out::PartialCleared { user_id }),
                Err(e) => {
                    (self.out)(Out::PartialCleared { user_id: user_id.clone() });
                    (self.out)(Out::Error { user_id: Some(user_id), message: format!("decode failed: {e}"), fatal: false });
                }
            }
            return;
        }
        let Some((user_id, epoch)) = self.partials.pop_front() else { return };
        let Some(s) = self.speakers.get(&user_id).filter(|s| s.epoch == epoch) else { return };
        let Some((from, to)) = s.seg.bounds() else { return };
        let start = s.frozen_at.unwrap_or(from);
        let freeze = (self.engine.windowed() && to - start > PARTIAL_WINDOW).then(|| {
            let cut = s.seg.quietest(start + PARTIAL_MIN_FREEZE, to - PARTIAL_KEEP);
            (cut, s.seg.slice(start, cut))
        });
        let tail = s.seg.slice(freeze.as_ref().map_or(start, |(cut, _)| *cut), to);

        // A failed partial isn't worth reporting; the final will say so.
        let started = Instant::now();
        let frozen = match freeze {
            Some((cut, clip)) => match self.engine.transcribe(&clip) {
                Ok(text) => Some((cut, text)),
                Err(_) => return,
            },
            None => None,
        };
        let text = self.engine.transcribe(&tail);
        self.last_run = started.elapsed();
        let Ok(text) = text else { return };

        let Some(s) = self.speakers.get_mut(&user_id) else { return };
        if let Some((cut, more)) = frozen {
            s.frozen_at = Some(cut);
            s.frozen = format!("{} {}", s.frozen, more.trim()).trim().to_string();
        }
        let clean = format!("{} {}", s.frozen, text.trim()).trim().to_string();
        if !clean.is_empty() && clean != s.last_partial {
            s.last_partial = clean.clone();
            (self.out)(Out::Partial { user_id, text: clean });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Transcribes a 16 kHz mono WAV with every ONNX model in a folder:
    /// `LAYER_TEST_WAV=path LAYER_TEST_MODELS=dir cargo test --release transcribes -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn transcribes() {
        let wav = std::fs::read(std::env::var("LAYER_TEST_WAV").expect("LAYER_TEST_WAV")).unwrap();
        let data = wav.windows(4).position(|w| w == b"data").expect("a data chunk") + 8;
        let audio: Vec<f32> = wav[data..].chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect();
        let dir = PathBuf::from(std::env::var("LAYER_TEST_MODELS").expect("LAYER_TEST_MODELS"));
        onnx::init(None).unwrap();
        for model in crate::models::Catalog::empty().list(&dir).into_iter().filter(|m| m.engine != "vosk") {
            let t = Instant::now();
            let engine = match model.engine.as_str() {
                "whisper" => whisper::Whisper::load(&model.path).map(ClipEngine::Whisper),
                "moonshine" => moonshine::Moonshine::load(&model.path).map(ClipEngine::Moonshine),
                _ => parakeet::Parakeet::load(&model.path).map(ClipEngine::Parakeet),
            };
            let mut engine = engine.unwrap_or_else(|e| panic!("{}: {e}", model.name));
            let loaded = t.elapsed();
            let t = Instant::now();
            let text = engine.transcribe(&audio).unwrap_or_else(|e| panic!("{}: {e}", model.name));
            println!("{:32} load {:>5} ms  decode {:>5} ms  {text}", model.name, loaded.as_millis(), t.elapsed().as_millis());
            assert!(!text.is_empty(), "{} heard nothing", model.name);
        }
    }
}
