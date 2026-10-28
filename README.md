# LayerKit

Code shared by the Layer apps (ChatterLayer, JammyLayer, BiscuitLayer), as
Rust crates.

| Crate | What |
|---|---|
| `layer-secrets` | Secrets (app keys, sign-ins) in the system keystore: Windows Credential Manager, the macOS Keychain, or the Secret Service on Linux, with a fallback to the app's files where there's none |
| `layer-speech` | Local speech-to-text: Moonshine, Parakeet, Whisper and Vosk engines, the microphone gate, and installed-model discovery |
| `layer-spotify` | Spotify: sign-in, what's playing, track lookup and search, and the player (queue, play, skip, previous, pause, resume, volume) |
| `layer-twitch` | Twitch: device-code sign-in for a channel and an optional bot, Helix calls, chat replies, channel points, and the EventSub socket |

## Using it

Apps depend on it by tag:

```toml
layer-secrets = { git = "https://github.com/ruptz/LayerKit", tag = "v0.5.0" }
layer-speech = { git = "https://github.com/ruptz/LayerKit", tag = "v0.5.0" }
layer-spotify = { git = "https://github.com/ruptz/LayerKit", tag = "v0.5.0" }
layer-twitch = { git = "https://github.com/ruptz/LayerKit", tag = "v0.5.0" }
```

### layer-secrets

Keep secrets out of the app's JSON files. `Vault::new("JammyLayer")` checks
once whether the keystore can be used. Reading a file, pass each secret
through `vault.adopt(name, &mut value)`: a value still in the file moves into
the keystore (rewrite the file when it returns true), an empty one is filled
from it. Writing, pass each through `vault.keep(name, &mut value)`, which
blanks the file's copy once the keystore has it. Without a keystore both leave
the value alone, so the file keeps it as before. JammyLayer's
`src-tauri/src/settings.rs` is the worked example.

On Linux it needs `libdbus-1-dev` and `pkg-config` to build.

### layer-speech

Name the app once at startup, so errors and overrides use its name:

```rust
layer_speech::init(layer_speech::App {
    name: "ChatterLayer",
    env_prefix: "CHATTERLAYER", // reads CHATTERLAYER_STT_THREADS, ..._ORT_LIB, ...
    dev_vendor: Some(Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("vendor")),
});
```

### layer-spotify

The app implements `Host` (its Spotify app's keys and redirect URI, the scopes
to ask for, where the sign-in is saved) and passes a `Config` (its name). It
opens `authorize_url()`, and hands the code its redirect receives to
`exchange_code()`. Ask for `CONTROL_SCOPE` alongside `READ_SCOPES` to change
playback; most player calls also need Spotify Premium, and say so when it's
missing. Anything not wrapped is one `spotify.call(method, path, body)` away.
JammyLayer's `src-tauri/src/spotify.rs` is the worked example; polling stays in
the app.

### layer-twitch

The app implements `Host` (its Twitch app's keys, where the sign-in is saved,
whether it needs events right now) and passes a `Config` (its name, scopes and
EventSub subscriptions). `Twitch::new` returns the client and a channel of
`Incoming` events; spawn `twitch.run()` for the socket. Anything not wrapped is
one `twitch.helix(seat, method, path, body)` away. JammyLayer's
`src-tauri/src/twitch.rs` is the worked example.

Each app keeps its own Spotify and Twitch sign-ins for now.

## Working on it beside an app

To work on LayerKit and an app together, check both out side by side and add
this to the app's `src-tauri/.cargo/config.toml` (gitignored):

```toml
[patch."https://github.com/ruptz/LayerKit"]
layer-speech = { path = "../../LayerKit/crates/layer-speech" }
layer-secrets = { path = "../../LayerKit/crates/layer-secrets" }
layer-spotify = { path = "../../LayerKit/crates/layer-spotify" }
layer-twitch = { path = "../../LayerKit/crates/layer-twitch" }
```

While the patch is on, Cargo drops LayerKit's `source` line from the app's
`Cargo.lock`. Don't commit that: tag LayerKit, bump the tag in the app, and
build once with the patch off so the lock pins the tag again.

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
