//! Turns a continuous stream of audio into utterances, for the engines that
//! need whole ones. Vosk decides for itself where an utterance ends; Whisper,
//! Moonshine and Parakeet transcribe a clip and stop, so something has to
//! decide what a clip is.
//!
//! Discord's own speaking-end signal is what usually closes one, as an explicit
//! flush. Trailing silence is the fallback, and matters because someone who
//! talks for a minute straight would otherwise be one clip no model can take.
//!
//! The speech test is deliberately cheap — frame energy against a slowly
//! tracked noise floor. It is not a VAD model; it only has to know whether
//! there's anything worth a decode. That matters: Whisper hallucinates
//! confidently on silence (a stray "Thank you." is the classic), so an
//! utterance that never held speech is thrown away, not transcribed.

const SAMPLE_RATE: usize = 16_000;
/// 20 ms — one Discord frame's worth.
const FRAME: usize = SAMPLE_RATE / 50;

/// About -50 dBFS. Below this nothing counts as speech however quiet the room,
/// which stops the tracker adapting its way down into treating hiss as speech.
const ABSOLUTE_FLOOR: f32 = 0.003;

/// About -34 dBFS. Energy alone can't tell a constant tone from constant hiss,
/// so a tracker that rises fast enough for a hissy mic would, over a long turn,
/// rise into the speech itself. Discord suppresses noise before we see the
/// audio, so a "noise floor" above this is almost certainly speech.
const NOISE_CEILING: f32 = 0.02;

/// How far above the tracked noise floor a frame has to be.
const SPEECH_MARGIN: f32 = 3.0;

const fn ms(n: usize) -> usize {
    SAMPLE_RATE * n / 1000
}

/// Trailing quiet that closes an utterance. 600 ms split sentences at
/// ordinary rhetorical pauses, and a segment cut mid-phrase loses the context
/// the model needs ("and so my fellow Americans" came back as "and so am I
/// fellow Americans"). 800 ms leaves real pauses alone.
const SILENCE: usize = ms(800);
/// Less speech than this and there is nothing worth transcribing.
const MIN_SPEECH: usize = ms(200);
/// A buffer this long with no speech in it is just noise.
const MAX_SILENT: usize = ms(3000);
/// Kept before the first speech frame.
const LEAD: usize = ms(200);
/// Kept after the last speech frame. Larger than the lead: a word ending in a
/// fricative trails off below the threshold while still being part of the
/// word, and "Americans" came back as "American" at 120 ms.
const TRAIL: usize = ms(300);

#[derive(Debug, PartialEq)]
pub enum Verdict {
    Final,
    /// Nothing in the buffer was speech; not worth a decode.
    Discard,
}

pub struct Segmenter {
    max_samples: usize,
    buf: Vec<f32>,
    analyzed: usize,
    speech: usize,
    first_speech: Option<usize>,
    last_speech: usize,
    /// RMS of each analysed frame, for finding a quiet place to cut.
    levels: Vec<f32>,
    /// Not reset between utterances: it belongs to the speaker's mic and room.
    noise: f32,
}

impl Segmenter {
    /// `max_seconds`: hard ceiling on one clip; Whisper can't encode past 30.
    pub fn new(max_seconds: usize) -> Self {
        Self {
            max_samples: SAMPLE_RATE * max_seconds,
            buf: Vec::with_capacity(SAMPLE_RATE * 4),
            analyzed: 0,
            speech: 0,
            first_speech: None,
            last_speech: 0,
            levels: Vec::with_capacity(200),
            noise: ABSOLUTE_FLOOR,
        }
    }

    pub fn reset(&mut self) {
        self.buf.clear();
        self.analyzed = 0;
        self.speech = 0;
        self.first_speech = None;
        self.last_speech = 0;
        self.levels.clear();
    }

    pub fn has_speech(&self) -> bool {
        self.speech >= MIN_SPEECH
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn push(&mut self, samples: &[f32]) {
        self.buf.extend_from_slice(samples);
        while self.analyzed + FRAME <= self.buf.len() {
            let start = self.analyzed;
            let frame = &self.buf[start..start + FRAME];
            let rms = (frame.iter().map(|x| x * x).sum::<f32>() / FRAME as f32).sqrt();
            self.levels.push(rms);
            if rms > ABSOLUTE_FLOOR.max(self.noise * SPEECH_MARGIN) {
                self.speech += FRAME;
                self.first_speech.get_or_insert(start);
                self.last_speech = start + FRAME;
            }
            // Minimum-statistics floor: fall towards any quieter frame, rise
            // only gradually. A symmetric average dragged the floor up through
            // long turns until quiet words stopped counting; 2%/frame settles
            // on a hissy mic in about a second, and NOISE_CEILING keeps it out
            // of the speech.
            self.noise = if rms < self.noise {
                self.noise * 0.5 + rms * 0.5
            } else {
                (self.noise * 1.02 + 1e-6).min(NOISE_CEILING)
            }
            .max(ABSOLUTE_FLOOR);
            self.analyzed = start + FRAME;
        }
    }

    /// What to do with the buffer right now, if anything.
    pub fn poll(&self) -> Option<Verdict> {
        let len = self.buf.len();
        if len == 0 {
            return None;
        }
        if !self.has_speech() {
            return (len >= MAX_SILENT).then_some(Verdict::Discard);
        }
        (len >= self.max_samples || len - self.last_speech >= SILENCE).then_some(Verdict::Final)
    }

    /// The speech in the buffer, with a short margin either side. Leading and
    /// trailing silence is pure decode cost, and for Whisper it is where
    /// hallucinations come from.
    pub fn view(&self) -> Option<Vec<f32>> {
        self.bounds().map(|(from, to)| self.slice(from, to))
    }

    /// Where `view` would cut, as sample offsets into the buffer.
    pub fn bounds(&self) -> Option<(usize, usize)> {
        if !self.has_speech() {
            return None;
        }
        let from = self.first_speech?.saturating_sub(LEAD);
        let to = self.buf.len().min(self.last_speech + TRAIL);
        (to > from).then_some((from, to))
    }

    pub fn slice(&self, from: usize, to: usize) -> Vec<f32> {
        self.buf[from..to].to_vec()
    }

    /// The quietest moment between two offsets, judged over 100 ms rather than
    /// one frame so a dip inside a word doesn't win: where a cut is least
    /// likely to split one.
    pub fn quietest(&self, from: usize, to: usize) -> usize {
        const SPAN: usize = 5;
        let (lo, hi) = (from / FRAME, (to / FRAME).min(self.levels.len()));
        if hi < lo + SPAN {
            return from;
        }
        let loudness = |i: usize| self.levels[i..i + SPAN].iter().sum::<f32>();
        let best = (lo..=hi - SPAN).min_by(|&a, &b| loudness(a).total_cmp(&loudness(b))).unwrap_or(lo);
        (best + SPAN / 2) * FRAME
    }

    /// `view`, then start a new utterance.
    pub fn take(&mut self) -> Option<Vec<f32>> {
        let clip = self.view();
        self.reset();
        clip
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(ms_len: usize, amp: f32) -> Vec<f32> {
        (0..ms(ms_len)).map(|i| amp * (i as f32 * 0.2).sin()).collect()
    }

    #[test]
    fn speech_then_silence_closes() {
        let mut s = Segmenter::new(20);
        s.push(&tone(600, 0.3));
        assert_eq!(s.poll(), None);
        s.push(&vec![0.0; ms(900)]);
        assert_eq!(s.poll(), Some(Verdict::Final));
        let clip = s.take().unwrap();
        assert!(clip.len() >= ms(600) && clip.len() <= ms(600) + TRAIL);
        assert!(s.is_empty());
    }

    #[test]
    fn silence_alone_is_discarded() {
        let mut s = Segmenter::new(20);
        s.push(&vec![0.001; ms(3100)]);
        assert_eq!(s.poll(), Some(Verdict::Discard));
        assert!(s.view().is_none());
    }

    #[test]
    fn quietest_finds_the_gap() {
        let mut s = Segmenter::new(20);
        s.push(&tone(1000, 0.3));
        s.push(&tone(300, 0.01));
        s.push(&tone(1000, 0.3));
        let cut = s.quietest(ms(200), ms(2100));
        assert!(cut >= ms(1000) && cut <= ms(1300), "cut at {} ms", cut * 1000 / SAMPLE_RATE);
    }
}
