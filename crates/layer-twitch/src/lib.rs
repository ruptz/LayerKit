//! The Twitch account for a Layer app: events arrive over one EventSub
//! WebSocket, and calls go out through Helix.
//!
//! Signing in uses Twitch's device code flow: the streamer types a short code
//! at twitch.tv/activate, so there's no redirect URL to register and nothing
//! for Twitch to call back into.
//!
//! Two accounts can be signed in. The channel's own is required: only the
//! broadcaster can run a channel points reward or refund it. A bot account is
//! optional and, when there is one, chat replies come from it instead.
//!
//! The socket is open only while the app says it needs it, so when nothing
//! does this costs nothing at all.
//!
//! Lifted out of JammyLayer. What's the app's own — its Twitch application's
//! keys, where the sign-in is saved, its scopes and which events it listens
//! for — comes in through `Host` and `Config`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc};
use tokio_tungstenite::tungstenite::Message;

const ID: &str = "https://id.twitch.tv/oauth2";
const HELIX: &str = "https://api.twitch.tv/helix";
const EVENTSUB: &str = "wss://eventsub.wss.twitch.tv/ws";

const DEVICE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";
/// Twitch asks apps to check their token is still valid once an hour.
const VALIDATE_EVERY: Duration = Duration::from_secs(60 * 60);
const BACKOFF_MAX: Duration = Duration::from_secs(60);

/// The Twitch application the streamer registered on Twitch's developer
/// console. The secret is optional: without one it's a Public client, whose
/// sign-in lasts 30 days.
#[derive(Clone, Default)]
pub struct Credentials {
    pub client_id: String,
    pub client_secret: String,
}

/// What the app keeps between runs. The channel's sign-in sits at the top
/// level; the optional bot account that replies in chat is nested.
#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct TwitchTokens {
    #[serde(flatten)]
    pub channel: SeatTokens,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bot: Option<SeatTokens>,
}

#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct SeatTokens {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub login: Option<String>,
}

/// The app's side: its keys, its storage, and whether it needs the socket.
pub trait Host: Send + Sync + 'static {
    fn credentials(&self) -> Credentials;
    /// Whether anything switched on needs events right now. The socket also
    /// needs credentials and a signed-in channel.
    fn wants_events(&self) -> bool;
    fn load_tokens(&self) -> TwitchTokens;
    fn save_tokens(&self, tokens: &TwitchTokens);
}

/// What the app asks Twitch for.
pub struct Config {
    /// "JammyLayer": the user agent, and log lines as `[jammylayer]`.
    pub app: &'static str,
    pub version: &'static str,
    /// Space-separated, as Twitch wants them.
    pub channel_scopes: &'static str,
    pub bot_scopes: &'static str,
    /// Subscribed to on every new socket, in order.
    pub subscriptions: &'static [Subscription],
}

/// One EventSub subscription.
pub struct Subscription {
    pub kind: &'static str,
    pub version: &'static str,
    pub condition: Condition,
    /// In the error the setup page shows: "couldn't listen to chat".
    pub label: &'static str,
    /// Without it the connection is down. Otherwise a refusal is logged and
    /// the rest carries on, as channel points do on a channel without
    /// Affiliate.
    pub required: bool,
}

pub enum Condition {
    /// `{ broadcaster_user_id }`
    Broadcaster,
    /// `{ broadcaster_user_id, user_id }`, both the channel: chat read as the
    /// channel itself.
    BroadcasterAsUser,
}

/// How far up the channel a viewer is.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Rank {
    Viewer,
    Subscriber,
    Vip,
    Moderator,
    Broadcaster,
}

#[derive(Clone)]
pub struct Viewer {
    pub id: String,
    pub login: String,
    pub name: String,
    pub rank: Rank,
}

pub enum Incoming {
    Chat { message_id: String, viewer: Viewer, text: String },
    Redemption { id: String, reward_id: String, viewer: Viewer, input: String },
    /// Any other subscription's notification, as Twitch sent it.
    Event { kind: String, event: Value },
    /// The connection or account changed; the setup page wants to know.
    Changed,
}

#[derive(Serialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Link {
    /// Not wanted: nothing needs it or there's no account.
    #[default]
    Off,
    Connecting,
    Live,
    /// Wanted, but down; retrying.
    Down,
}

/// Which of the two sign-ins.
#[derive(Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Seat {
    Channel,
    Bot,
}

impl Seat {
    pub fn parse(value: &str) -> Self {
        if value == "bot" { Seat::Bot } else { Seat::Channel }
    }

    fn index(self) -> usize {
        self as usize
    }
}

#[derive(Default)]
struct Account {
    refresh_token: Option<String>,
    access_token: Option<String>,
    expires_at: Option<Instant>,
    validated_at: Option<Instant>,
    user_id: Option<String>,
    login: Option<String>,
    auth_error: Option<String>,
}

impl Account {
    fn connected(&self) -> bool {
        self.refresh_token.is_some() && self.user_id.is_some()
    }

    fn saved(&self) -> Option<SeatTokens> {
        self.connected().then(|| SeatTokens {
            refresh_token: self.refresh_token.clone(),
            user_id: self.user_id.clone(),
            login: self.login.clone(),
        })
    }

    fn restored(saved: Option<SeatTokens>) -> Self {
        let saved = saved.unwrap_or_default();
        Account { refresh_token: saved.refresh_token, user_id: saved.user_id, login: saved.login, ..Default::default() }
    }
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    configured: bool,
    client_id: String,
    has_secret: bool,
    channel: SeatStatus,
    bot: SeatStatus,
    link: Link,
    link_error: Option<String>,
    pairing: Option<Pairing>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SeatStatus {
    connected: bool,
    login: Option<String>,
    error: Option<String>,
}

/// A sign-in waiting for its code to be entered on Twitch.
#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Pairing {
    seat: Seat,
    user_code: String,
    /// Twitch's activation page with the code already filled in.
    verification_uri: String,
    expires_at: u64,
}

pub enum RewardError {
    /// Deleted on Twitch's side, or made by some other app.
    Missing,
    Other(String),
}

pub struct Twitch {
    http: reqwest::Client,
    host: Arc<dyn Host>,
    config: Config,
    accounts: [Mutex<Account>; 2],
    pairing: Mutex<Option<Pairing>>,
    /// Bumped whenever a pairing starts or is abandoned, so an older one
    /// still polling Twitch knows to stop.
    pairing_generation: AtomicU64,
    refreshing: tokio::sync::Mutex<()>,
    link: Mutex<(Link, Option<String>)>,
    /// Account or switches changed: the socket loop re-checks what it should be doing.
    wake: Notify,
    /// Bumped on every sign-in and sign-out, so a socket subscribed for one
    /// account doesn't carry on for the next.
    generation: AtomicU64,
    incoming: mpsc::UnboundedSender<Incoming>,
}

impl Twitch {
    pub fn new(host: Arc<dyn Host>, config: Config) -> (Arc<Self>, mpsc::UnboundedReceiver<Incoming>) {
        let saved = host.load_tokens();
        let (incoming, rx) = mpsc::unbounded_channel();
        let twitch = Arc::new(Self {
            http: reqwest::Client::builder()
                .user_agent(format!("{}/{}", config.app, config.version))
                .build()
                .expect("an HTTP client"),
            host,
            config,
            accounts: [Mutex::new(Account::restored(Some(saved.channel))), Mutex::new(Account::restored(saved.bot))],
            pairing: Mutex::new(None),
            pairing_generation: AtomicU64::new(0),
            refreshing: tokio::sync::Mutex::new(()),
            link: Mutex::new((Link::Off, None)),
            wake: Notify::new(),
            generation: AtomicU64::new(0),
            incoming,
        });
        (twitch, rx)
    }

    pub fn configured(&self) -> bool {
        !self.host.credentials().client_id.is_empty()
    }

    fn account(&self, seat: Seat) -> MutexGuard<'_, Account> {
        self.accounts[seat.index()].lock().unwrap()
    }

    fn scopes(&self, seat: Seat) -> &'static str {
        match seat {
            Seat::Channel => self.config.channel_scopes,
            Seat::Bot => self.config.bot_scopes,
        }
    }

    fn log(&self, message: &str) {
        eprintln!("[{}] {message}", self.config.app.to_lowercase());
    }

    /// The channel's own sign-in, which everything but replies needs.
    pub fn connected(&self) -> bool {
        self.account(Seat::Channel).connected()
    }

    pub fn user_id(&self) -> Option<String> {
        self.account(Seat::Channel).user_id.clone()
    }

    pub fn status(&self) -> Status {
        let credentials = self.host.credentials();
        let (link, link_error) = self.link.lock().unwrap().clone();
        let seat = |seat: Seat| {
            let a = self.account(seat);
            SeatStatus { connected: a.connected(), login: a.login.clone(), error: a.auth_error.clone() }
        };
        Status {
            configured: !credentials.client_id.is_empty(),
            client_id: credentials.client_id,
            has_secret: !credentials.client_secret.is_empty(),
            channel: seat(Seat::Channel),
            bot: seat(Seat::Bot),
            link,
            link_error,
            pairing: self.pairing.lock().unwrap().clone(),
        }
    }

    /// Tells the setup page to redraw.
    pub fn changed(&self) {
        let _ = self.incoming.send(Incoming::Changed);
    }

    fn set_link(&self, link: Link, error: Option<String>) {
        let mut current = self.link.lock().unwrap();
        if *current != (link, error.clone()) {
            *current = (link, error);
            drop(current);
            self.changed();
        }
    }

    /// The switches changed; open or close the socket to match.
    pub fn reconsider(&self) {
        self.wake.notify_one();
    }

    fn wanted(&self) -> bool {
        self.host.wants_events() && self.configured() && self.connected()
    }

    // ------------------------------------------------------------ authorization

    async fn token_request(&self, form: &[(&str, &str)]) -> Result<Value, (String, u16)> {
        let Credentials { client_id: id, client_secret: secret } = self.host.credentials();
        let mut form = form.to_vec();
        form.push(("client_id", &id));
        // A Public client has no secret; its tokens just don't last as long.
        if !secret.is_empty() {
            form.push(("client_secret", &secret));
        }
        let resp = self
            .http
            .post(format!("{ID}/token"))
            .form(&form)
            .send()
            .await
            .map_err(|_| ("Couldn't reach Twitch.".to_string(), 0))?;
        let status = resp.status().as_u16();
        let body: Value = resp.json().await.unwrap_or(Value::Null);
        if !(200..300).contains(&status) {
            let message = body["message"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("token request failed ({status})"));
            return Err((message, status));
        }
        Ok(body)
    }

    /// Who the token belongs to. Also the hourly check Twitch asks for.
    async fn validate(&self, token: &str) -> Result<(String, String), String> {
        let resp = self
            .http
            .get(format!("{ID}/validate"))
            .header("Authorization", format!("OAuth {token}"))
            .send()
            .await
            .map_err(|_| "Couldn't reach Twitch.".to_string())?;
        if !resp.status().is_success() {
            return Err("Twitch no longer accepts the saved sign-in.".into());
        }
        let body: Value = resp.json().await.map_err(|_| "Twitch sent something unexpected.".to_string())?;
        let id = body["user_id"].as_str().unwrap_or_default().to_owned();
        let login = body["login"].as_str().unwrap_or_default().to_owned();
        Ok((id, login))
    }

    fn remember(&self) {
        let tokens = TwitchTokens {
            channel: self.account(Seat::Channel).saved().unwrap_or_default(),
            bot: self.account(Seat::Bot).saved(),
        };
        self.host.save_tokens(&tokens);
    }

    /// Asks Twitch for a code to show the streamer, then waits in the
    /// background for them to approve it.
    pub async fn start_pairing(self: &Arc<Self>, seat: Seat) -> Result<Pairing, String> {
        if seat == Seat::Bot && !self.connected() {
            return Err("Connect your channel first.".into());
        }
        let client_id = self.host.credentials().client_id;
        if client_id.is_empty() {
            return Err("Save your Twitch client ID first.".into());
        }
        let resp = self
            .http
            .post(format!("{ID}/device"))
            .form(&[("client_id", client_id.as_str()), ("scopes", self.scopes(seat))])
            .send()
            .await
            .map_err(|_| "Couldn't reach Twitch.".to_string())?;
        let status = resp.status().as_u16();
        let body: Value = resp.json().await.unwrap_or(Value::Null);
        if !(200..300).contains(&status) {
            return Err(match body["message"].as_str() {
                Some("invalid client") => "Twitch doesn't recognise that client ID.".into(),
                Some(message) => format!("Twitch refused: {message}"),
                None => format!("Twitch returned {status}"),
            });
        }
        let text = |k: &str| body[k].as_str().unwrap_or_default().to_owned();
        let device_code = text("device_code");
        let expires = Duration::from_secs(body["expires_in"].as_u64().unwrap_or(1800));
        let interval = Duration::from_secs(body["interval"].as_u64().unwrap_or(5).max(1));
        let pairing = Pairing {
            seat,
            user_code: text("user_code"),
            verification_uri: text("verification_uri"),
            expires_at: now_ms() + expires.as_millis() as u64,
        };
        let generation = self.pairing_generation.fetch_add(1, Ordering::SeqCst) + 1;
        *self.pairing.lock().unwrap() = Some(pairing.clone());
        self.account(seat).auth_error = None;
        self.changed();
        let twitch = self.clone();
        tokio::spawn(async move {
            twitch.await_pairing(seat, device_code, interval, Instant::now() + expires, generation).await;
        });
        Ok(pairing)
    }

    pub fn cancel_pairing(&self) {
        self.pairing_generation.fetch_add(1, Ordering::SeqCst);
        *self.pairing.lock().unwrap() = None;
        self.changed();
    }

    async fn await_pairing(
        &self,
        seat: Seat,
        device_code: String,
        mut interval: Duration,
        deadline: Instant,
        generation: u64,
    ) {
        let current = || self.pairing_generation.load(Ordering::SeqCst) == generation;
        let outcome = loop {
            tokio::time::sleep(interval).await;
            if !current() {
                return;
            }
            if Instant::now() > deadline {
                break Err("The code expired before it was entered. Connect again.".to_string());
            }
            match self
                .token_request(&[("grant_type", DEVICE_GRANT), ("device_code", &device_code), ("scopes", self.scopes(seat))])
                .await
            {
                Ok(granted) => break Ok(granted),
                Err((message, _)) => match message.as_str() {
                    "authorization_pending" => {}
                    "slow_down" => interval += Duration::from_secs(5),
                    "access_denied" => break Err("Twitch sign-in was declined.".into()),
                    "invalid device code" => break Err("The code expired. Connect again.".into()),
                    // A dropped connection mid-wait isn't the end of it.
                    "Couldn't reach Twitch." => {}
                    _ => break Err(format!("Twitch sign-in failed: {message}")),
                },
            }
        };
        if !current() {
            return;
        }
        let outcome = match outcome {
            Ok(granted) => self.signed_in(seat, &granted).await,
            Err(e) => Err(e),
        };
        *self.pairing.lock().unwrap() = None;
        if let Err(e) = outcome {
            self.account(seat).auth_error = Some(e);
        }
        self.changed();
    }

    async fn signed_in(&self, seat: Seat, granted: &Value) -> Result<(), String> {
        let token = granted["access_token"].as_str().unwrap_or_default().to_owned();
        let (user_id, login) = self.validate(&token).await?;
        // Easy to do by accident: twitch.tv/activate opens in the browser
        // that's already signed in as the streamer.
        if seat == Seat::Bot && self.user_id().as_deref() == Some(user_id.as_str()) {
            return Err(format!(
                "That code was approved as {login}, your channel. Approve it while signed in to Twitch as the bot account; a private window makes that easy."
            ));
        }
        *self.account(seat) = Account {
            refresh_token: granted["refresh_token"].as_str().map(str::to_owned),
            access_token: Some(token),
            expires_at: Some(Instant::now() + expires_in(granted)),
            validated_at: Some(Instant::now()),
            user_id: Some(user_id),
            login: Some(login),
            auth_error: None,
        };
        self.remember();
        if seat == Seat::Channel {
            self.generation.fetch_add(1, Ordering::SeqCst);
            self.wake.notify_one();
        }
        Ok(())
    }

    /// Forgetting the channel forgets the bot too: it only ever spoke there.
    pub fn disconnect(&self, seat: Seat) {
        *self.account(seat) = Account::default();
        if seat == Seat::Channel {
            *self.account(Seat::Bot) = Account::default();
            self.generation.fetch_add(1, Ordering::SeqCst);
            self.wake.notify_one();
        }
        self.remember();
        self.changed();
    }

    async fn access_token(&self, seat: Seat) -> Result<String, String> {
        if !self.configured() {
            return Err("Twitch client credentials are not set".into());
        }
        let _one_at_a_time = self.refreshing.lock().await;
        let (refresh, cached, checked) = {
            let account = self.account(seat);
            let Some(refresh) = account.refresh_token.clone() else {
                return Err("No Twitch account connected".into());
            };
            let fresh = account.expires_at.is_some_and(|e| e > Instant::now() + Duration::from_secs(60));
            let checked = account.validated_at.is_some_and(|v| v.elapsed() < VALIDATE_EVERY);
            (refresh, account.access_token.clone().filter(|_| fresh), checked)
        };
        if let Some(token) = cached
            && (checked || self.validate(&token).await.is_ok())
        {
            if !checked {
                self.account(seat).validated_at = Some(Instant::now());
            }
            return Ok(token);
        }
        match self.token_request(&[("grant_type", "refresh_token"), ("refresh_token", &refresh)]).await {
            Ok(granted) => {
                let token = granted["access_token"].as_str().unwrap_or_default().to_owned();
                // A refreshed token is checked as it's issued, which covers
                // the hourly validation too.
                let (user_id, login) = self.validate(&token).await?;
                {
                    let mut account = self.account(seat);
                    account.access_token = Some(token.clone());
                    account.expires_at = Some(Instant::now() + expires_in(&granted));
                    account.validated_at = Some(Instant::now());
                    account.user_id = Some(user_id);
                    account.login = Some(login);
                    account.auth_error = None;
                    if let Some(rotated) = granted["refresh_token"].as_str() {
                        account.refresh_token = Some(rotated.to_owned());
                    }
                }
                self.remember();
                Ok(token)
            }
            Err((message, _)) => {
                // Only a refresh token Twitch calls invalid is gone for good.
                // Anything else — a missing secret, Twitch having a bad
                // minute — keeps the sign-in, so it works again once fixed.
                let message = refresh_problem(&message);
                if message.is_none() {
                    *self.account(seat) = Account {
                        auth_error: Some("Twitch no longer accepts the saved sign-in. Connect the account again.".into()),
                        ..Default::default()
                    };
                    self.remember();
                } else {
                    self.account(seat).auth_error = message.clone();
                }
                self.changed();
                Err(message.unwrap_or_else(|| "Twitch sign-in expired".into()))
            }
        }
    }

    /// Any Helix call, as `seat`, with one retry on a token Twitch has
    /// stopped accepting. `path` starts with `/` and carries its own query.
    pub async fn helix(
        &self,
        seat: Seat,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<reqwest::Response, String> {
        let url = format!("{HELIX}{path}");
        let client_id = self.host.credentials().client_id;
        let send = |token: String| {
            let mut req = self.http.request(method.clone(), &url).bearer_auth(token).header("Client-Id", &client_id);
            if let Some(body) = &body {
                req = req.json(body);
            }
            req.send()
        };
        let unreachable = |_| "Couldn't reach Twitch".to_string();
        let mut resp = send(self.access_token(seat).await?).await.map_err(unreachable)?;
        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            self.account(seat).access_token = None;
            resp = send(self.access_token(seat).await?).await.map_err(unreachable)?;
        }
        Ok(resp)
    }

    // ------------------------------------------------------------------- chat

    /// Said by the bot if there is one, otherwise by the broadcaster,
    /// threaded under the message it answers. A bot that can't post says why
    /// in the setup page rather than quietly falling back.
    pub async fn say(&self, text: &str, reply_to: Option<&str>) {
        let Some(channel) = self.user_id() else { return };
        let bot = {
            let bot = self.account(Seat::Bot);
            bot.user_id.clone().filter(|_| bot.connected())
        };
        let (seat, sender) = match bot {
            Some(bot) => (Seat::Bot, bot),
            None => (Seat::Channel, channel.clone()),
        };
        let mut body = json!({ "broadcaster_id": channel, "sender_id": sender, "message": text });
        if let Some(parent) = reply_to {
            body["reply_parent_message_id"] = json!(parent);
        }
        let problem = match self.helix(seat, reqwest::Method::POST, "/chat/messages", Some(body)).await {
            Ok(resp) if resp.status().is_success() => {
                let body: Value = resp.json().await.unwrap_or(Value::Null);
                // Twitch answers 200 for a message its filters dropped.
                match body["data"][0]["is_sent"].as_bool() {
                    Some(false) => Some(format!(
                        "Twitch dropped a reply: {}",
                        body["data"][0]["drop_reason"]["message"].as_str().unwrap_or("no reason given")
                    )),
                    _ => None,
                }
            }
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body: Value = resp.json().await.unwrap_or(Value::Null);
                Some(format!("Couldn't post in chat ({status}: {})", body["message"].as_str().unwrap_or("no reason given")))
            }
            Err(e) => Some(e),
        };
        if let Some(problem) = &problem {
            self.log(problem);
        }
        let mut account = self.account(seat);
        if account.connected() && account.auth_error != problem {
            account.auth_error = problem;
            drop(account);
            self.changed();
        }
    }

    // ---------------------------------------------------------- channel points

    /// A reward viewers redeem with text, which this app then owns: Twitch lets
    /// only the app that made a reward pause it or refund its redemptions.
    pub async fn create_reward(&self, title: &str, cost: u32, prompt: &str) -> Result<String, String> {
        let me = self.user_id().ok_or("No Twitch account connected")?;
        let body = json!({
            "title": title,
            "cost": cost,
            "prompt": prompt,
            "is_user_input_required": true,
            // Left in the queue, so a redemption that can't be honoured is refunded.
            "should_redemptions_skip_request_queue": false,
        });
        let resp = self
            .helix(Seat::Channel, reqwest::Method::POST, &format!("/channel_points/custom_rewards?broadcaster_id={me}"), Some(body))
            .await?;
        let status = resp.status().as_u16();
        let body: Value = resp.json().await.unwrap_or(Value::Null);
        if !(200..300).contains(&status) {
            return Err(reward_error(status, &body));
        }
        body["data"][0]["id"].as_str().map(str::to_owned).ok_or_else(|| "Twitch sent something unexpected".into())
    }

    pub async fn update_reward(&self, reward_id: &str, changes: Value) -> Result<(), RewardError> {
        let me = self.user_id().ok_or(RewardError::Other("No Twitch account connected".into()))?;
        let resp = self
            .helix(
                Seat::Channel,
                reqwest::Method::PATCH,
                &format!("/channel_points/custom_rewards?broadcaster_id={me}&id={}", encode(reward_id)),
                Some(changes),
            )
            .await
            .map_err(RewardError::Other)?;
        let status = resp.status().as_u16();
        if (200..300).contains(&status) {
            return Ok(());
        }
        let body: Value = resp.json().await.unwrap_or(Value::Null);
        // 403 is a reward some other app made, which this one can't touch.
        if status == 404 || (status == 403 && !body["message"].as_str().unwrap_or_default().contains("affiliate")) {
            return Err(RewardError::Missing);
        }
        Err(RewardError::Other(reward_error(status, &body)))
    }

    /// FULFILLED once it's been honoured, CANCELED to hand the points back.
    pub async fn settle(&self, reward_id: &str, redemption_id: &str, status: &str) {
        let Some(me) = self.user_id() else { return };
        let path = format!(
            "/channel_points/custom_rewards/redemptions?broadcaster_id={me}&reward_id={}&id={}",
            encode(reward_id),
            encode(redemption_id)
        );
        match self.helix(Seat::Channel, reqwest::Method::PATCH, &path, Some(json!({ "status": status }))).await {
            Ok(resp) if !resp.status().is_success() => {
                self.log(&format!("could not mark redemption {status}: {}", resp.status()));
            }
            Err(e) => self.log(&format!("could not mark redemption {status}: {e}")),
            _ => {}
        }
    }

    // ----------------------------------------------------------------- socket

    /// Runs for the life of the app, connected only while the app wants it.
    pub async fn run(self: Arc<Self>) {
        let mut failures = 0u32;
        loop {
            if !self.wanted() {
                self.set_link(Link::Off, None);
                self.wake.notified().await;
                continue;
            }
            self.set_link(Link::Connecting, None);
            let mut url = EVENTSUB.to_string();
            let mut fresh = true;
            let error = loop {
                match self.session(&url, fresh, &mut failures).await {
                    // Twitch moving us: subscriptions follow to the new URL.
                    Ok(Some(next)) => {
                        url = next;
                        fresh = false;
                    }
                    Ok(None) => break None,
                    Err(e) => break Some(e),
                }
            };
            let Some(error) = error else { continue };
            self.log(&format!("Twitch connection: {error}"));
            self.set_link(Link::Down, Some(error));
            let wait = Duration::from_secs(2u64.saturating_pow(failures.min(6))).min(BACKOFF_MAX);
            failures += 1;
            tokio::select! {
                _ = tokio::time::sleep(wait) => {}
                _ = self.wake.notified() => {}
            }
        }
    }

    /// One connection. Ok(Some(url)) to reconnect there, Ok(None) when it's no
    /// longer wanted.
    async fn session(&self, url: &str, fresh: bool, failures: &mut u32) -> Result<Option<String>, String> {
        let generation = self.generation.load(Ordering::SeqCst);
        let (mut ws, _) = tokio_tungstenite::connect_async(url)
            .await
            .map_err(|e| format!("couldn't connect ({e})"))?;
        let mut keepalive = Duration::from_secs(10);
        // Generous: Twitch's keepalive is a ceiling on its silence, and a
        // notification counts too.
        let silence = |keepalive: Duration| keepalive + Duration::from_secs(5);
        let mut welcomed = false;
        loop {
            let frame = tokio::select! {
                frame = tokio::time::timeout(silence(keepalive), ws.next()) => frame,
                _ = self.wake.notified() => {
                    if self.wanted() && self.generation.load(Ordering::SeqCst) == generation { continue }
                    let _ = ws.close(None).await;
                    return Ok(None);
                }
            };
            let text = match frame {
                Err(_) => return Err("Twitch went quiet".into()),
                Ok(None) => return Err("Twitch closed the connection".into()),
                Ok(Some(Err(e))) => return Err(e.to_string()),
                Ok(Some(Ok(Message::Text(text)))) => text,
                Ok(Some(Ok(Message::Close(frame)))) => {
                    return Err(frame.map(|f| f.reason.to_string()).unwrap_or_else(|| "closed".into()));
                }
                Ok(Some(Ok(_))) => continue,
            };
            let Ok(message) = serde_json::from_str::<Value>(text.as_str()) else { continue };
            let payload = &message["payload"];
            match message["metadata"]["message_type"].as_str().unwrap_or_default() {
                "session_welcome" => {
                    if let Some(secs) = payload["session"]["keepalive_timeout_seconds"].as_u64() {
                        keepalive = Duration::from_secs(secs);
                    }
                    if fresh && !welcomed {
                        let session = payload["session"]["id"].as_str().unwrap_or_default().to_owned();
                        self.subscribe(&session).await?;
                    }
                    welcomed = true;
                    *failures = 0;
                    self.set_link(Link::Live, None);
                }
                "session_reconnect" => {
                    if let Some(next) = payload["session"]["reconnect_url"].as_str() {
                        return Ok(Some(next.to_owned()));
                    }
                }
                "notification" => self.notification(&message["metadata"]["subscription_type"], &payload["event"]),
                "revocation" => {
                    let kind = payload["subscription"]["type"].as_str().unwrap_or_default();
                    return Err(format!("Twitch revoked {kind}; the sign-in may need renewing"));
                }
                _ => {}
            }
        }
    }

    async fn subscribe(&self, session: &str) -> Result<(), String> {
        let me = self.user_id().ok_or("No Twitch account connected")?;
        let transport = json!({ "method": "websocket", "session_id": session });
        for sub in self.config.subscriptions {
            let condition = match sub.condition {
                Condition::Broadcaster => json!({ "broadcaster_user_id": me }),
                Condition::BroadcasterAsUser => json!({ "broadcaster_user_id": me, "user_id": me }),
            };
            let body = json!({ "type": sub.kind, "version": sub.version, "condition": condition, "transport": transport });
            let resp = self.helix(Seat::Channel, reqwest::Method::POST, "/eventsub/subscriptions", Some(body)).await?;
            if resp.status().is_success() {
                continue;
            }
            if !sub.required {
                self.log(&format!("{} unavailable: {}", sub.label, resp.status()));
                continue;
            }
            let status = resp.status().as_u16();
            let body: Value = resp.json().await.unwrap_or(Value::Null);
            return Err(format!(
                "couldn't listen to {} ({status}: {})",
                sub.label,
                body["message"].as_str().unwrap_or("no reason given")
            ));
        }
        Ok(())
    }

    fn notification(&self, kind: &Value, event: &Value) {
        let text = |v: &Value| v.as_str().unwrap_or_default().to_owned();
        let viewer = |prefix: &str| Viewer {
            id: text(&event[format!("{prefix}_id")]),
            login: text(&event[format!("{prefix}_login")]),
            name: text(&event[format!("{prefix}_name")]),
            rank: Rank::Viewer,
        };
        let incoming = match kind.as_str().unwrap_or_default() {
            "channel.chat.message" => {
                let mut viewer = viewer("chatter_user");
                viewer.rank = rank(&event["badges"]);
                Incoming::Chat { message_id: text(&event["message_id"]), viewer, text: text(&event["message"]["text"]) }
            }
            "channel.channel_points_custom_reward_redemption.add" => Incoming::Redemption {
                id: text(&event["id"]),
                reward_id: text(&event["reward"]["id"]),
                viewer: viewer("user"),
                input: text(&event["user_input"]),
            },
            "" => return,
            other => Incoming::Event { kind: other.to_owned(), event: event.clone() },
        };
        let _ = self.incoming.send(incoming);
    }
}

/// The highest badge a chatter wears.
fn rank(badges: &Value) -> Rank {
    let Some(badges) = badges.as_array() else { return Rank::Viewer };
    badges
        .iter()
        .map(|b| match b["set_id"].as_str().unwrap_or_default() {
            "broadcaster" => Rank::Broadcaster,
            "moderator" | "lead_moderator" => Rank::Moderator,
            "vip" => Rank::Vip,
            "subscriber" | "founder" => Rank::Subscriber,
            _ => Rank::Viewer,
        })
        .max()
        .unwrap_or(Rank::Viewer)
}

/// None when Twitch has revoked the refresh token; otherwise what to tell
/// the streamer, the sign-in kept.
fn refresh_problem(message: &str) -> Option<String> {
    let lower = message.to_lowercase();
    if lower.contains("invalid refresh token") {
        return None;
    }
    Some(if lower.contains("secret") || lower.contains("invalid client") {
        "Twitch wants your app's client secret to keep you signed in: your app is Confidential. Add the secret in step 1."
            .into()
    } else {
        format!("Couldn't renew the Twitch sign-in: {message}")
    })
}

fn reward_error(status: u16, body: &Value) -> String {
    let message = body["message"].as_str().unwrap_or_default();
    if message.to_lowercase().contains("affiliate") || message.to_lowercase().contains("partner") {
        return "Channel points need a Twitch Affiliate or Partner channel".into();
    }
    if message.contains("DUPLICATE") || message.to_lowercase().contains("same title") {
        return "Another reward on the channel already has that title".into();
    }
    if message.is_empty() { format!("Twitch returned {status}") } else { message.to_owned() }
}

fn expires_in(granted: &Value) -> Duration {
    Duration::from_secs(granted["expires_in"].as_u64().unwrap_or(3600))
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
    fn rank_is_the_highest_badge() {
        let badges = json!([{ "set_id": "subscriber" }, { "set_id": "moderator" }, { "set_id": "bits" }]);
        assert_eq!(rank(&badges), Rank::Moderator);
        assert_eq!(rank(&json!([{ "set_id": "founder" }])), Rank::Subscriber);
        assert_eq!(rank(&Value::Null), Rank::Viewer);
    }

    #[test]
    fn only_a_revoked_refresh_token_signs_out() {
        assert!(refresh_problem("Invalid refresh token").is_none());
        assert!(refresh_problem("missing client secret").unwrap().contains("Confidential"));
        assert!(refresh_problem("bad gateway").unwrap().contains("bad gateway"));
    }

    #[test]
    fn encodes_query_values() {
        assert_eq!(encode("a b&c/é"), "a%20b%26c%2F%C3%A9");
        assert_eq!(encode("Az09-_.~"), "Az09-_.~");
    }
}
