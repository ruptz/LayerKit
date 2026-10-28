//! Whisper's log-mel front end.
//!
//! Whisper is the only engine here that needs features computed on our side:
//! Moonshine consumes raw samples, and Parakeet ships its own extractor as an
//! ONNX graph. This has to match what the model was trained on exactly — the
//! reference is `whisper.audio.log_mel_spectrogram`, whose filterbank comes
//! from `librosa.filters.mel(sr, n_fft, n_mels)`.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

// Hz to mel on the Slaney scale (librosa's default, htk=False): linear below
// 1 kHz, logarithmic above it.
const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;

fn log_step() -> f64 {
    6.4f64.ln() / 27.0
}

fn hz_to_mel(hz: f64) -> f64 {
    if hz < MIN_LOG_HZ { hz / F_SP } else { MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / log_step() }
}

fn mel_to_hz(mel: f64) -> f64 {
    if mel < MIN_LOG_MEL { F_SP * mel } else { MIN_LOG_HZ * (log_step() * (mel - MIN_LOG_MEL)).exp() }
}

/// Triangular mel filterbank with Slaney area normalisation, n_mels x bins,
/// row-major.
fn mel_filterbank(sample_rate: f64, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sample_rate / 2.0);
    // n_mels + 2 band edges: each filter spans edge[m] .. edge[m+2].
    let edges: Vec<f64> =
        (0..n_mels + 2).map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)).collect();
    let mut filters = vec![0f32; n_mels * bins];
    for m in 0..n_mels {
        let (lo, mid, hi) = (edges[m], edges[m + 1], edges[m + 2]);
        // Equal area per filter rather than equal peak, so wide high-frequency
        // filters don't dominate.
        let enorm = 2.0 / (hi - lo);
        for b in 0..bins {
            let f = b as f64 * sample_rate / n_fft as f64;
            let w = ((f - lo) / (mid - lo)).min((hi - f) / (hi - mid));
            if w > 0.0 {
                filters[m * bins + b] = (w * enorm) as f32;
            }
        }
    }
    filters
}

pub struct WhisperFeatures {
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    bins: usize,
    window: Vec<f64>,
    filters: Vec<f32>,
    fft: Arc<dyn Fft<f64>>,
}

impl WhisperFeatures {
    pub fn new(sample_rate: usize, n_fft: usize, hop: usize, n_mels: usize) -> Self {
        // torch.hann_window defaults to periodic, not symmetric.
        let window = (0..n_fft).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos()).collect();
        Self {
            n_fft,
            hop,
            n_mels,
            bins: n_fft / 2 + 1,
            window,
            filters: mel_filterbank(sample_rate as f64, n_fft, n_mels),
            // An exact 400-point transform: zero-padding to 512 would shift
            // every bin and the filters would read the wrong frequencies.
            fft: FftPlanner::new().plan_fft_forward(n_fft),
        }
    }

    /// n_mels x n_frames, mel-major, for a [1, n_mels, n_frames] tensor.
    pub fn compute(&self, audio: &[f32], n_frames: usize) -> Vec<f32> {
        let (n_fft, hop, n_mels, bins) = (self.n_fft, self.hop, self.n_mels, self.bins);
        let half = n_fft / 2;
        let n = audio.len();
        let mut out = vec![0f32; n_mels * n_frames];

        // Whisper zero-pads to 30 s before the STFT, so every frame starting
        // past the real audio is an identical all-zero column, filled in
        // afterwards instead of transformed thousands of times.
        let real = n_frames.min((n + half).div_ceil(hop));

        let mut buf = vec![Complex::new(0.0, 0.0); n_fft];
        let mut power = vec![0f64; bins];
        let mut gmax = f32::NEG_INFINITY;
        for t in 0..real {
            let start = (t * hop) as isize - half as isize;
            for (i, slot) in buf.iter_mut().enumerate() {
                // torch.stft(center=True) reflects at the start.
                let idx = (start + i as isize).unsigned_abs();
                let x = if idx < n { audio[idx] as f64 } else { 0.0 };
                *slot = Complex::new(x * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            for (p, c) in power.iter_mut().zip(&buf) {
                *p = c.norm_sqr();
            }
            for m in 0..n_mels {
                let row = &self.filters[m * bins..(m + 1) * bins];
                let acc: f64 = row.iter().zip(&power).map(|(&f, &p)| f as f64 * p).sum();
                let v = acc.max(1e-10).log10() as f32;
                out[m * n_frames + t] = v;
                gmax = gmax.max(v);
            }
        }

        // Clamp to 80 dB below the loudest bin, then map roughly onto [-1, 1].
        // A zero frame sits at log10(1e-10) = -10, so the floor is what the
        // padding columns become.
        if gmax == f32::NEG_INFINITY {
            gmax = -10.0;
        }
        let floor = (gmax - 8.0).max(-10.0);
        let pad = (floor + 4.0) / 4.0;
        for m in 0..n_mels {
            let row = &mut out[m * n_frames..(m + 1) * n_frames];
            for v in &mut row[..real] {
                *v = (v.max(floor) + 4.0) / 4.0;
            }
            row[real..].fill(pad);
        }
        out
    }
}
