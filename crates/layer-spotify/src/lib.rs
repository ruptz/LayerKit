//! The Spotify account for a Layer app: signing in, what's playing, and the
//! player — queue, skip, pause, volume, search and play.
//!
//! Signing in is Spotify's authorization code flow, with the streamer's own
//! Spotify app: the app opens `authorize_url`, Spotify sends the browser back
//! to the app's redirect URI, and the app hands the code to `exchange_code`.
//!
//! Lifted out of JammyLayer. What's the app's own — its Spotify app's keys and
//! redirect, the scopes it asks for, where the sign-in is saved — comes in
//! through `Host`. Polling and what an overlay is shown stay in the app.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const ACCOUNTS: &str = "https://accounts.spotify.com";
const API: &str = "https://api.spotify.com/v1";

/// Reading what's playing.
pub const READ_SCOPES: &str = "user-read-currently-playing user-read-playback-state";
/// Changing it: queue, skip, pause, volume, play.
pub const CONTROL_SCOPE: &str = "user-modify-playback-state";

/// How long a "Connect Spotify" link stays usable.
const STATE_LIFETIME: Duration = Duration::from_secs(10 * 60);

/// The Spotify app the streamer registered on Spotify's developer dashboard.
#[derive(Clone, Default)]
pub struct Credentials {
    pub client_id: String,
    pub client_secret: String,
    /// Has to be listed on that app exactly. Spotify accepts only HTTPS or the
    /// literal loopback address; "localhost" is refused.
    pub redirect_uri: String,
}

/// What the app keeps between runs.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Tokens {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub obtained_at: Option<String>,
}

/// The app's side: its keys, what it asks for, and its storage.
pub trait Host: Send + Sync + 'static {
    fn credentials(&self) -> Credentials;
    /// Space-separated, for the next sign-in.
    fn scopes(&self) -> String;
    fn load_tokens(&self) -> Tokens;
    fn save_tokens(&self, tokens: &Tokens);
}

pub struct Config {
    /// "JammyLayer": the user agent.
    pub app: &'static str,
    pub version: &'static str,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Item {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub title: String,
    pub subtitle: String,
    pub context: String,
    pub duration_ms: u64,
    pub art_url: Option<String>,
    pub explicit: bool,
}

/// What's playing, as of one call.
#[derive(Default, Debug)]
pub struct NowPlaying {
    pub playing: bool,
    pub item: Option<Item>,
    pub progress_ms: u64,
    pub device: Option<String>,
}

/// A track to queue or play.
#[derive(Clone, Debug)]
pub struct Track {
    pub id: String,
    pub title: String,
    pub artist: String,
    pub art_url: Option<String>,
    pub duration_ms: u64,
    pub explicit: bool,
}

pub struct ApiError {
    pub message: String,
    /// Set when Spotify said how long to back off.
    pub retry_after: Option<Duration>,
}

impl From<String> for ApiError {
    fn from(message: String) -> Self {
        Self { message, retry_after: None }
    }
}

#[derive(Default)]
struct Account {
    refresh_token: Option<String>,
    access_token: Option<String>,
    expires_at: Option<Instant>,
    /// What the account granted, as of the last token. An account connected
    /// before the app asked for more lacks it until reconnected.
    scope: Option<String>,
    auth_error: Option<String>,
}

pub struct Spotify {
    http: reqwest::Client,
    host: Arc<dyn Host>,
    account: Mutex<Account>,
    /// "Connect Spotify" links handed out and not yet used, with when they lapse.
    pending: Mutex<HashMap<String, Instant>>,
    /// One refresh at a time, however many requests want a token.
    refreshing: tokio::sync::Mutex<()>,
}

impl Spotify {
    pub fn new(host: Arc<dyn Host>, config: Config) -> Arc<Self> {
        let refresh_token = host.load_tokens().refresh_token;
        Arc::new(Self {
            http: reqwest::Client::builder()
                .user_agent(format!("{}/{}", config.app, config.version))
                .build()
                .expect("an HTTP client"),
            host,
            account: Mutex::new(Account { refresh_token, ..Default::default() }),
            pending: Mutex::new(HashMap::new()),
            refreshing: tokio::sync::Mutex::new(()),
        })
    }

    pub fn configured(&self) -> bool {
        let c = self.host.credentials();
        !c.client_id.is_empty() && !c.client_secret.is_empty()
    }

    pub fn connected(&self) -> bool {
        self.account.lock().unwrap().refresh_token.is_some()
    }

    /// Why the saved sign-in stopped working, if it did.
    pub fn auth_error(&self) -> Option<String> {
        self.account.lock().unwrap().auth_error.clone()
    }

    /// Whether the player can be changed. None until a token has said either way.
    pub fn can_control(&self) -> Option<bool> {
        let account = self.account.lock().unwrap();
        account.refresh_token.as_ref()?;
        let scope = account.scope.as_deref()?;
        Some(scope.split(' ').any(|s| s == CONTROL_SCOPE))
    }

    // ------------------------------------------------------------ authorization

    pub fn authorize_url(&self) -> String {
        let mut bytes = [0u8; 16];
        getrandom::fill(&mut bytes).expect("randomness for the login state");
        let state: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        {
            let mut pending = self.pending.lock().unwrap();
            let now = Instant::now();
            pending.retain(|_, expires| *expires > now);
            pending.insert(state.clone(), now + STATE_LIFETIME);
        }
        let credentials = self.host.credentials();
        let scopes = self.host.scopes();
        let query = [
            ("response_type", "code"),
            ("client_id", credentials.client_id.as_str()),
            ("scope", scopes.as_str()),
            ("redirect_uri", credentials.redirect_uri.as_str()),
            ("state", state.as_str()),
        ]
        .iter()
        .map(|(k, v)| format!("{k}={}", encode(v)))
        .collect::<Vec<_>>()
        .join("&");
        format!("{ACCOUNTS}/authorize?{query}")
    }

    /// Each link works once, and only within its lifetime.
    pub fn consume_state(&self, state: &str) -> bool {
        self.pending
            .lock()
            .unwrap()
            .remove(state)
            .is_some_and(|expires| expires >= Instant::now())
    }

    async fn token_request(&self, form: &[(&str, &str)]) -> Result<Value, (String, Option<String>)> {
        let Credentials { client_id: id, client_secret: secret, .. } = self.host.credentials();
        let resp = self
            .http
            .post(format!("{ACCOUNTS}/api/token"))
            .basic_auth(id, Some(secret))
            .form(form)
            .send()
            .await
            .map_err(|_| ("Couldn't reach Spotify.".to_string(), None))?;
        let status = resp.status();
        let body: Value = resp.json().await.unwrap_or(Value::Null);
        if !status.is_success() {
            let code = body["error"].as_str().map(str::to_owned);
            let message = body["error_description"]
                .as_str()
                .or(code.as_deref())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("token request failed ({status})"));
            return Err((message, code));
        }
        Ok(body)
    }

    fn remember(&self, refresh_token: &str) {
        self.host.save_tokens(&Tokens {
            refresh_token: Some(refresh_token.to_owned()),
            obtained_at: Some(now_ms().to_string()),
        });
    }

    pub async fn exchange_code(&self, code: &str) -> Result<(), String> {
        let redirect = self.host.credentials().redirect_uri;
        let granted = self
            .token_request(&[
                ("grant_type", "authorization_code"),
                ("code", code),
                ("redirect_uri", &redirect),
            ])
            .await
            .map_err(|(message, _)| message)?;
        let refresh = granted["refresh_token"].as_str().unwrap_or_default().to_owned();
        {
            let mut account = self.account.lock().unwrap();
            account.access_token = granted["access_token"].as_str().map(str::to_owned);
            account.expires_at = Some(Instant::now() + expires_in(&granted));
            account.scope = granted["scope"].as_str().map(str::to_owned);
            account.refresh_token = Some(refresh.clone());
            account.auth_error = None;
        }
        self.remember(&refresh);
        Ok(())
    }

    pub fn disconnect(&self) {
        *self.account.lock().unwrap() = Account::default();
        self.host.save_tokens(&Tokens::default());
    }

    async fn access_token(&self) -> Result<String, String> {
        if !self.configured() {
            return Err("Spotify client credentials are not set".into());
        }
        let _one_at_a_time = self.refreshing.lock().await;
        let refresh = {
            let account = self.account.lock().unwrap();
            let Some(refresh) = account.refresh_token.clone() else {
                return Err("No Spotify account connected".into());
            };
            if let (Some(token), Some(expires)) = (&account.access_token, account.expires_at)
                && expires > Instant::now() + Duration::from_secs(60)
            {
                return Ok(token.clone());
            }
            refresh
        };
        match self
            .token_request(&[("grant_type", "refresh_token"), ("refresh_token", &refresh)])
            .await
        {
            Ok(granted) => {
                let token = granted["access_token"].as_str().unwrap_or_default().to_owned();
                let rotated = granted["refresh_token"].as_str().filter(|r| *r != refresh);
                let mut account = self.account.lock().unwrap();
                account.access_token = Some(token.clone());
                account.expires_at = Some(Instant::now() + expires_in(&granted));
                if let Some(scope) = granted["scope"].as_str() {
                    account.scope = Some(scope.to_owned());
                }
                account.auth_error = None;
                // A rotated refresh token has to be saved, or the next start
                // signs in with one Spotify has already retired.
                if let Some(rotated) = rotated {
                    account.refresh_token = Some(rotated.to_owned());
                    drop(account);
                    self.remember(rotated);
                }
                Ok(token)
            }
            Err((message, code)) => {
                if code.as_deref() == Some("invalid_grant") {
                    *self.account.lock().unwrap() = Account {
                        auth_error: Some(
                            "Spotify rejected the saved authorization. Connect the account again."
                                .into(),
                        ),
                        ..Default::default()
                    };
                    self.host.save_tokens(&Tokens::default());
                }
                Err(message)
            }
        }
    }

    /// Any Web API call, with one retry on a token Spotify has stopped
    /// accepting. `path` starts with `/` and carries its own query.
    pub async fn call(&self, method: reqwest::Method, path: &str, body: Option<Value>) -> Result<reqwest::Response, String> {
        let url = format!("{API}{path}");
        let send = |token: String| {
            let req = self.http.request(method.clone(), &url).bearer_auth(token);
            match &body {
                Some(body) => req.json(body),
                // The player POSTs carry everything in the query, but Spotify
                // still wants a length.
                None => req.header(reqwest::header::CONTENT_LENGTH, "0"),
            }
            .send()
        };
        let unreachable = |_| "Couldn't reach Spotify".to_string();
        let mut resp = send(self.access_token().await?).await.map_err(unreachable)?;
        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            // Revoked or expired early: force one refresh and retry once.
            self.account.lock().unwrap().access_token = None;
            resp = send(self.access_token().await?).await.map_err(unreachable)?;
        }
        Ok(resp)
    }

    // -------------------------------------------------------------- now playing

    pub async fn now_playing(&self) -> Result<NowPlaying, ApiError> {
        let resp = self
            .call(reqwest::Method::GET, "/me/player/currently-playing?additional_types=track,episode", None)
            .await?;
        match resp.status().as_u16() {
            429 => {
                let secs = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(5);
                Err(ApiError {
                    message: "Spotify rate limit reached".into(),
                    retry_after: Some(Duration::from_secs(secs + 1)),
                })
            }
            // Connected, but nothing is loaded in any player.
            204 => Ok(NowPlaying::default()),
            status if !(200..300).contains(&status) => {
                Err(format!("Spotify API returned {status}").into())
            }
            _ => {
                let body: Value = resp
                    .json()
                    .await
                    .map_err(|_| "Spotify sent something unexpected".to_string())?;
                Ok(normalize(&body))
            }
        }
    }

    // ------------------------------------------------------------------ tracks

    pub async fn track(&self, id: &str) -> Result<Track, String> {
        let resp = self.call(reqwest::Method::GET, &format!("/tracks/{id}?market=from_token"), None).await?;
        match resp.status().as_u16() {
            400 | 404 => return Err("that track doesn't exist on Spotify".into()),
            status if !(200..300).contains(&status) => return Err(refused(status, resp).await),
            _ => {}
        }
        let body: Value = resp.json().await.map_err(|_| "Spotify sent something unexpected".to_string())?;
        if body["is_playable"] == false {
            return Err("that track isn't playable in this country".into());
        }
        Ok(track_from(&body))
    }

    /// Tracks matching what someone said, best first, playable ones only.
    pub async fn search(&self, query: &str, limit: u8) -> Result<Vec<Track>, String> {
        // No market=from_token: that needs the user-read-private scope, which
        // nothing else here does, and the search is no worse without it.
        let path = format!("/search?type=track&limit={}&q={}", limit.clamp(1, 50), encode(query));
        let resp = self.call(reqwest::Method::GET, &path, None).await?;
        if !resp.status().is_success() {
            return Err(refused(resp.status().as_u16(), resp).await);
        }
        let body: Value = resp.json().await.map_err(|_| "Spotify sent something unexpected".to_string())?;
        Ok(body["tracks"]["items"]
            .as_array()
            .map(|items| items.iter().filter(|t| t["is_playable"] != false).map(track_from).collect())
            .unwrap_or_default())
    }

    // ------------------------------------------------------------------ player

    pub async fn add_to_queue(&self, track_id: &str) -> Result<(), String> {
        let uri = encode(&format!("spotify:track:{track_id}"));
        let resp = self.call(reqwest::Method::POST, &format!("/me/player/queue?uri={uri}"), None).await?;
        player_result(resp, QUEUE).await
    }

    /// Whether a track is still waiting in Spotify's own queue.
    pub async fn queued(&self, track_id: &str) -> Result<bool, String> {
        let resp = self.call(reqwest::Method::GET, "/me/player/queue", None).await?;
        if !resp.status().is_success() {
            return Err(refused(resp.status().as_u16(), resp).await);
        }
        let body: Value = resp.json().await.map_err(|_| "Spotify sent something unexpected".to_string())?;
        Ok(body["queue"].as_array().is_some_and(|q| q.iter().any(|t| t["id"] == track_id)))
    }

    /// Plays a track now, in place of whatever was playing, and checks that
    /// it did.
    ///
    /// Sent to the device that's playing — or, with nothing playing, the one
    /// used last — as the track's album starting at that track, which is what
    /// Spotify's own apps send. A bare track ("uris") is accepted by the
    /// Windows app with a 204 and then not played at all.
    pub async fn play(&self, track_id: &str) -> Result<(), String> {
        let device = self.device().await?;
        let track = self.get(&format!("/tracks/{track_id}?market=from_token")).await;
        let uri = format!("spotify:track:{track_id}");
        let body = match track["album"]["uri"].as_str() {
            Some(album) => json!({ "context_uri": album, "offset": { "uri": uri } }),
            None => json!({ "uris": [uri] }),
        };
        let path = format!("/me/player/play?device_id={}", encode(&device));
        player_result(self.call(reqwest::Method::PUT, &path, Some(body)).await?, PLAYBACK).await?;
        // A moment for the player to switch, then ask what it's playing.
        tokio::time::sleep(Duration::from_millis(900)).await;
        let now = self.get("/me/player/currently-playing").await;
        let title = track["name"].as_str().unwrap_or_default();
        let started = now["is_playing"] == true
            && (now["item"]["id"] == track_id || (!title.is_empty() && now["item"]["name"].as_str() == Some(title)));
        if started {
            Ok(())
        } else {
            Err("Spotify took the request but didn't start the song. Try again, or play it once in Spotify itself.".into())
        }
    }

    /// The device that's playing, or the one used last.
    async fn device(&self) -> Result<String, String> {
        if let Some(id) = self.get("/me/player").await["device"]["id"].as_str() {
            return Ok(id.to_owned());
        }
        self.get("/me/player/devices").await["devices"]
            .as_array()
            .and_then(|d| d.first())
            .and_then(|d| d["id"].as_str())
            .map(str::to_owned)
            .ok_or_else(|| "Spotify isn't open anywhere. Open it on your PC or phone, then try again.".into())
    }

    /// A GET's body, or null when there isn't one to read.
    async fn get(&self, path: &str) -> Value {
        match self.call(reqwest::Method::GET, path, None).await {
            Ok(resp) if resp.status().is_success() => resp.json().await.unwrap_or(Value::Null),
            _ => Value::Null,
        }
    }

    pub async fn skip(&self) -> Result<(), String> {
        let resp = self.call(reqwest::Method::POST, "/me/player/next", None).await?;
        player_result(resp, PLAYBACK).await
    }

    pub async fn previous(&self) -> Result<(), String> {
        let resp = self.call(reqwest::Method::POST, "/me/player/previous", None).await?;
        player_result(resp, PLAYBACK).await
    }

    pub async fn pause(&self) -> Result<(), String> {
        let resp = self.call(reqwest::Method::PUT, "/me/player/pause", None).await?;
        player_result(resp, PLAYBACK).await
    }

    pub async fn resume(&self) -> Result<(), String> {
        let resp = self.call(reqwest::Method::PUT, "/me/player/play", None).await?;
        player_result(resp, PLAYBACK).await
    }

    /// The active player's volume, 0–100. None when the device doesn't
    /// allow it to be set, as most phones don't.
    pub async fn volume(&self) -> Result<Option<u8>, String> {
        let resp = self.call(reqwest::Method::GET, "/me/player", None).await?;
        match resp.status().as_u16() {
            204 => return Err("nothing is playing on Spotify".into()),
            status if !(200..300).contains(&status) => return Err(refused(status, resp).await),
            _ => {}
        }
        let body: Value = resp.json().await.map_err(|_| "Spotify sent something unexpected".to_string())?;
        if body["device"]["supports_volume"] == false {
            return Ok(None);
        }
        Ok(body["device"]["volume_percent"].as_u64().map(|v| v.min(100) as u8))
    }

    pub async fn set_volume(&self, percent: u8) -> Result<(), String> {
        let path = format!("/me/player/volume?volume_percent={}", percent.min(100));
        let resp = self.call(reqwest::Method::PUT, &path, None).await?;
        player_result(resp, PLAYBACK).await
    }
}

fn expires_in(granted: &Value) -> Duration {
    Duration::from_secs(granted["expires_in"].as_u64().unwrap_or(3600))
}

/// What a player call was for, for its refusals: what Premium is needed for,
/// and what reconnecting would allow.
struct Purpose {
    premium: &'static str,
    allow: &'static str,
}

const QUEUE: Purpose = Purpose { premium: "to queue songs", allow: "song requests" };
const PLAYBACK: Purpose = Purpose { premium: "to control playback", allow: "playback control" };

/// The player endpoints' refusals, in words a streamer can act on.
async fn player_result(resp: reqwest::Response, purpose: Purpose) -> Result<(), String> {
    let status = resp.status().as_u16();
    if (200..300).contains(&status) {
        return Ok(());
    }
    let body: Value = resp.json().await.unwrap_or(Value::Null);
    let reason = body["error"]["reason"].as_str().unwrap_or_default();
    let message = body["error"]["message"].as_str().unwrap_or_default();
    Err(refusal(status, reason, message, &purpose))
}

fn refusal(status: u16, reason: &str, message: &str, purpose: &Purpose) -> String {
    match (status, reason) {
        (403, "PREMIUM_REQUIRED") => format!("Spotify Premium is needed {}", purpose.premium),
        (404, _) | (_, "NO_ACTIVE_DEVICE") => "nothing is playing on Spotify".into(),
        (403, _) if message.contains("scope") => format!("reconnect Spotify to allow {}", purpose.allow),
        (429, _) => "Spotify rate limit reached".into(),
        _ if message.is_empty() => format!("Spotify returned {status}"),
        _ => format!("Spotify returned {status}: {message}"),
    }
}

/// Any other refusal, with Spotify's own reason when it gives one, so a
/// problem nobody foresaw can still be read.
async fn refused(status: u16, resp: reqwest::Response) -> String {
    let body: Value = resp.json().await.unwrap_or(Value::Null);
    match body["error"]["message"].as_str().filter(|m| !m.is_empty()) {
        Some(message) => format!("Spotify returned {status}: {message}"),
        None => format!("Spotify returned {status}"),
    }
}

fn artists(item: &Value) -> String {
    item["artists"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x["name"].as_str()).collect::<Vec<_>>().join(", "))
        .unwrap_or_default()
}

fn track_from(body: &Value) -> Track {
    Track {
        id: body["id"].as_str().unwrap_or_default().to_owned(),
        title: body["name"].as_str().unwrap_or_default().to_owned(),
        artist: artists(body),
        art_url: pick_image(&body["album"]["images"]),
        duration_ms: body["duration_ms"].as_u64().unwrap_or(0),
        explicit: body["explicit"].as_bool().unwrap_or(false),
    }
}

/// Around 300px keeps the art crisp at overlay sizes without pulling a 640px
/// JPEG every track.
fn pick_image(images: &Value) -> Option<String> {
    images
        .as_array()?
        .iter()
        .min_by_key(|i| (i["width"].as_i64().unwrap_or(0) - 300).abs())
        .and_then(|i| i["url"].as_str())
        .map(str::to_owned)
}

fn normalize(body: &Value) -> NowPlaying {
    let device = body["device"]["name"].as_str().map(str::to_owned);
    let playing = body["is_playing"].as_bool().unwrap_or(false);
    let item = &body["item"];
    if item.is_null() {
        return NowPlaying { playing, device, ..Default::default() };
    }
    let episode = item["type"] == "episode";
    let images = if episode {
        if item["images"].is_array() { &item["images"] } else { &item["show"]["images"] }
    } else {
        &item["album"]["images"]
    };
    let text = |v: &Value| v.as_str().unwrap_or_default().to_owned();
    let subtitle = if episode { text(&item["show"]["name"]) } else { artists(item) };
    let id = [&item["id"], &item["uri"], &item["name"]]
        .into_iter()
        .find_map(|v| v.as_str().filter(|s| !s.is_empty()))
        .unwrap_or_default()
        .to_owned();
    NowPlaying {
        playing,
        progress_ms: body["progress_ms"].as_u64().unwrap_or(0),
        device,
        item: Some(Item {
            id,
            kind: text(&item["type"]),
            title: text(&item["name"]),
            subtitle,
            context: text(if episode { &item["show"]["publisher"] } else { &item["album"]["name"] }),
            duration_ms: item["duration_ms"].as_u64().unwrap_or(0),
            art_url: pick_image(images),
            explicit: item["explicit"].as_bool().unwrap_or(false),
        }),
    }
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// Percent-encoding for query values.
pub fn encode(value: &str) -> String {
    value
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_a_track_and_an_episode() {
        let track = normalize(&json!({
            "is_playing": true,
            "progress_ms": 1200,
            "device": { "name": "PC" },
            "item": {
                "id": "abc", "type": "track", "name": "Song", "duration_ms": 180000, "explicit": true,
                "artists": [{ "name": "A" }, { "name": "B" }],
                "album": { "name": "Album", "images": [{ "width": 640, "url": "big" }, { "width": 300, "url": "mid" }] },
            },
        }));
        let item = track.item.unwrap();
        assert!(track.playing);
        assert_eq!((item.subtitle.as_str(), item.context.as_str(), item.art_url.as_deref()), ("A, B", "Album", Some("mid")));

        let episode = normalize(&json!({
            "item": { "type": "episode", "uri": "spotify:episode:1", "name": "Ep", "show": { "name": "Show", "publisher": "Pub" } },
        }));
        let item = episode.item.unwrap();
        assert_eq!((item.id.as_str(), item.subtitle.as_str(), item.context.as_str()), ("spotify:episode:1", "Show", "Pub"));

        assert!(normalize(&json!({ "is_playing": false, "item": null })).item.is_none());
    }

    #[test]
    fn refusals_say_what_to_do() {
        assert_eq!(refusal(403, "PREMIUM_REQUIRED", "", &QUEUE), "Spotify Premium is needed to queue songs");
        assert_eq!(refusal(403, "PREMIUM_REQUIRED", "", &PLAYBACK), "Spotify Premium is needed to control playback");
        assert_eq!(refusal(404, "", "", &PLAYBACK), "nothing is playing on Spotify");
        assert_eq!(refusal(403, "", "Insufficient client scope", &PLAYBACK), "reconnect Spotify to allow playback control");
    }
}
