//! Voice gate for the local microphone.
//!
//! Discord runs its own voice detection and sends nothing while someone is
//! quiet, so everything downstream expects audio only while there's speech,
//! followed by a clean end. A microphone sends audio all the time; this gate
//! makes it behave like Discord. That matters for more than tidiness: given
//! long runs of room tone, Whisper-family models invent text ("Thank you.")
//! that would go straight on stream.

const RATE: usize = 16_000;
/// 20 ms, the same frame Discord uses.
const FRAME: usize = 320;

/// Speech has to stand this far above the room's noise floor (~10 dB).
const OVER_FLOOR: f64 = 3.2;
/// Loud frames in a row before it counts as speech, so a desk knock doesn't.
const ONSET_FRAMES: u32 = 3;
/// Audio kept from before speech was detected, so the first syllable survives.
const PRE_ROLL_MS: usize = 300;

/// The sensitivity slider runs 1 (only loud, close speech) to 10 (whispers).
/// The default, 5, is about -44 dBFS: well under normal speech into a headset
/// or desk mic. Each step is 3 dB, so the ends are -32 and -59 dBFS.
fn min_rms_for(sensitivity: f64) -> f64 {
    200.0 * 2f64.powf((5.0 - sensitivity.clamp(1.0, 10.0)) / 2.0)
}

fn frames_for(ms: f64) -> u32 {
    (ms / 1000.0 * RATE as f64 / FRAME as f64).ceil() as u32
}

pub struct MicGate {
    rest: Vec<i16>,
    /// Recent quiet frames, replayed when speech starts.
    pre_roll: std::collections::VecDeque<Vec<i16>>,
    hang_frames: u32,
    min_rms: f64,
    floor: f64,
    speaking: bool,
    loud_run: u32,
    quiet_run: u32,
}

pub struct Gated {
    /// Audio to recognise.
    pub voiced: Vec<Vec<i16>>,
    /// An utterance finished somewhere in this chunk.
    pub ended: bool,
}

impl MicGate {
    pub fn new(sensitivity: f64, hang_ms: f64) -> Self {
        let min_rms = min_rms_for(sensitivity);
        Self {
            rest: Vec::new(),
            pre_roll: Default::default(),
            hang_frames: frames_for(hang_ms),
            min_rms,
            floor: min_rms / OVER_FLOOR,
            speaking: false,
            loud_run: 0,
            quiet_run: 0,
        }
    }

    /// Retune mid-stream, from the sliders. What it learned about the room is
    /// kept.
    pub fn configure(&mut self, sensitivity: f64, hang_ms: f64) {
        self.hang_frames = frames_for(hang_ms);
        self.min_rms = min_rms_for(sensitivity);
    }

    /// 16 kHz mono.
    pub fn push(&mut self, pcm: &[i16]) -> Gated {
        self.rest.extend_from_slice(pcm);
        let all = std::mem::take(&mut self.rest);
        let pre_roll_frames = frames_for(PRE_ROLL_MS as f64) as usize;
        let mut voiced = Vec::new();
        let mut ended = false;
        let mut chunks = all.chunks_exact(FRAME);
        for frame in &mut chunks {
            let loud = self.is_loud(frame);
            if !self.speaking {
                self.pre_roll.push_back(frame.to_vec());
                if self.pre_roll.len() > pre_roll_frames {
                    self.pre_roll.pop_front();
                }
                self.loud_run = if loud { self.loud_run + 1 } else { 0 };
                if self.loud_run >= ONSET_FRAMES {
                    self.speaking = true;
                    self.quiet_run = 0;
                    voiced.extend(self.pre_roll.drain(..));
                }
                continue;
            }
            voiced.push(frame.to_vec());
            self.quiet_run = if loud { 0 } else { self.quiet_run + 1 };
            if self.quiet_run >= self.hang_frames {
                self.speaking = false;
                self.loud_run = 0;
                ended = true;
            }
        }
        self.rest = chunks.remainder().to_vec();
        Gated { voiced, ended }
    }

    fn is_loud(&mut self, frame: &[i16]) -> bool {
        let rms = (frame.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / frame.len() as f64).sqrt();
        let loud = rms > self.min_rms.max(self.floor * OVER_FLOOR);
        // Follow the room: down quickly when it goes quiet, up slowly, and far
        // more slowly while someone talks so their voice isn't learned as
        // noise. A fan switching on still gets absorbed.
        let rate = if rms < self.floor { 0.2 } else if loud { 0.002 } else { 0.02 };
        self.floor += (rms - self.floor) * rate;
        loud
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_on_speech_and_closes_after_the_hang() {
        let mut g = MicGate::new(5.0, 700.0);
        let quiet = vec![0i16; FRAME * 10];
        let loud: Vec<i16> = (0..FRAME * 20).map(|i| if i % 2 == 0 { 3000 } else { -3000 }).collect();
        assert!(g.push(&quiet).voiced.is_empty());
        let on = g.push(&loud);
        // Pre-roll included: more than the loud frames alone.
        assert!(on.voiced.len() > 18 && !on.ended);
        let off = g.push(&vec![0i16; FRAME * 40]);
        assert!(off.ended);
    }
}
