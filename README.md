# LayerKit

Code shared by the Layer apps (ChatterLayer, JammyLayer, BiscuitLayer), as
Rust crates.

| Crate | What |
|---|---|
| `layer-speech` | Local speech-to-text: Moonshine, Parakeet, Whisper and Vosk engines, the microphone gate, and installed-model discovery |

## Using it

Apps depend on it by tag:

```toml
layer-speech = { git = "https://github.com/ruptz/LayerKit", tag = "v0.1.0" }
```

and name themselves once at startup, so errors and overrides use the app's
name:

```rust
layer_speech::init(layer_speech::App {
    name: "ChatterLayer",
    env_prefix: "CHATTERLAYER", // reads CHATTERLAYER_STT_THREADS, ..._ORT_LIB, ...
    dev_vendor: Some(Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("vendor")),
});
```

To work on LayerKit and an app together, check both out side by side and add
this to the app's `src-tauri/.cargo/config.toml` (gitignored):

```toml
[patch."https://github.com/ruptz/LayerKit"]
layer-speech = { path = "../../LayerKit/crates/layer-speech" }
```

## Testing

```bash
cargo test
# Transcribe a 16 kHz mono WAV with every ONNX model in a folder:
LAYER_TEST_WAV=some.wav LAYER_TEST_MODELS=../ChatterLayer/models \
LAYER_ORT_LIB=../ChatterLayer/vendor/runtime/onnxruntime/onnxruntime.dll \
  cargo test --release transcribes -- --ignored --nocapture
```

## Licence

MIT.
