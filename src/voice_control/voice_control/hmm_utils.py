"""
Standalone HMM utilities for voice recognition (no external project imports).

Pipeline:
  signal → preemphasis → Hamming frames → FFT → Mel filterbank → log → DCT
         → MFCC (13 coeff) → VQ global (K=256) → discrete sequence
         → HMM Bakis Forward log-likelihood → best word

Vocabulary trained with 5 speakers × 15 samples × data augmentation (9 variants).
"""

import numpy as np

# ─── Audio constants ──────────────────────────────────────────────
FS        = 16000   # Hz
FRAME_LEN = 320     # samples (~20 ms)
HOP       = 128     # samples (~8 ms)
ALPHA     = 0.95    # preemphasis coefficient

# ─── MFCC constants ───────────────────────────────────────────────
N_MFCC    = 13
N_MELS    = 26
N_FFT     = 512
N_STATES  = 6
N_SYMBOLS = 256

PALABRAS = [
    "one", "two", "racks", "rollers", "map",
    "stop", "forward", "left", "right", "continue",
]


# ─────────────────────────────────────────────────────────────────
# 1. Preemphasis + framing
# ─────────────────────────────────────────────────────────────────

def apply_preemphasis(signal, alpha=ALPHA):
    return np.concatenate([[signal[0]], signal[1:] - alpha * signal[:-1]])


def hamming_frames(signal, frame_len=FRAME_LEN, hop=HOP):
    window   = np.hamming(frame_len)
    n        = len(signal)
    n_frames = max(1, 1 + (n - frame_len) // hop)
    frames   = np.zeros((n_frames, frame_len), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        chunk = signal[start:start + frame_len]
        frames[i, :len(chunk)] = chunk * window[:len(chunk)]
    return frames


# ─────────────────────────────────────────────────────────────────
# 2. Voice Activity Detection
# ─────────────────────────────────────────────────────────────────

def detect_endpoints(signal, frame_len=FRAME_LEN, hop=HOP, fs=FS):
    """Detect speech start/end using adaptive energy threshold.

    Returns (start_sample, end_sample). Falls back to (0, len(signal))
    when no speech is detected or the segment would be too short.
    """
    n        = len(signal)
    n_frames = max(1, 1 + (n - frame_len) // hop)
    energy   = np.zeros(n_frames)

    for i in range(n_frames):
        s     = i * hop
        frame = signal[s:s + frame_len]
        energy[i] = np.sum(frame ** 2) / max(len(frame), 1)

    e_thresh = 0.02 * np.max(energy)
    voiced   = np.where(energy > e_thresh)[0]

    if len(voiced) == 0:
        return 0, n

    # 4-frame margin so word onset/offset are not clipped
    first_frame = max(0, voiced[0] - 4)
    last_frame  = min(n_frames - 1, voiced[-1] + 4)

    start_sample = first_frame * hop
    end_sample   = min(last_frame * hop + frame_len, n)

    # Require at least 0.1 s of detected speech
    if end_sample - start_sample < int(0.1 * fs):
        return 0, n

    return start_sample, end_sample


# ─────────────────────────────────────────────────────────────────
# 3. Mel filterbank
# ─────────────────────────────────────────────────────────────────

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, fs=FS):
    low_mel  = _hz_to_mel(0)
    high_mel = _hz_to_mel(fs / 2)
    mel_pts  = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts   = _mel_to_hz(mel_pts)
    bins     = np.floor((n_fft + 1) * hz_pts / fs).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        if mid > lo:
            fbank[m - 1, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fbank[m - 1, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return fbank


# ─────────────────────────────────────────────────────────────────
# 4. MFCC extraction
# ─────────────────────────────────────────────────────────────────

def extract_mfcc(signal, fs=FS, n_mfcc=N_MFCC, n_mels=N_MELS,
                 n_fft=N_FFT, frame_len=FRAME_LEN, hop=HOP):
    """Extract MFCC features from a 1-D float32 signal.

    Returns:
        (n_frames, n_mfcc) float32 array
    """
    signal = apply_preemphasis(signal.astype(np.float32))
    frames = hamming_frames(signal, frame_len, hop)

    if len(frames) == 0:
        return np.zeros((0, n_mfcc), dtype=np.float32)

    n_frames = len(frames)
    padded   = np.zeros((n_frames, n_fft), dtype=np.float32)
    copy_len = min(frame_len, n_fft)
    padded[:, :copy_len] = frames[:, :copy_len]

    spectrum = np.abs(np.fft.rfft(padded, n=n_fft)) ** 2

    fbank   = mel_filterbank(n_mels, n_fft, fs)
    mel_e   = spectrum @ fbank.T
    mel_e   = np.maximum(mel_e, 1e-10)
    log_mel = np.log(mel_e)

    j   = np.arange(n_mels)
    k   = np.arange(n_mfcc).reshape(-1, 1)
    dct = np.cos(np.pi * k * (2 * j + 1) / (2 * n_mels))
    mfcc = log_mel @ dct.T

    return mfcc.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# 5. VQ quantization
# ─────────────────────────────────────────────────────────────────

def quantize(mfcc_seq, centroids):
    """Assign each MFCC frame to the nearest centroid.

    Args:
        mfcc_seq:  (T, D)
        centroids: (K, D)

    Returns:
        (T,) int32 indices in [0, K-1]
    """
    diff  = mfcc_seq[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)
    return np.argmin(dists, axis=1).astype(np.int32)


# ─────────────────────────────────────────────────────────────────
# 6. HMM Bakis (Left-to-Right)
# ─────────────────────────────────────────────────────────────────

def _logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    out   = np.log(np.sum(np.exp(a - a_max), axis=axis) + 1e-300)
    if axis is not None:
        out += np.squeeze(a_max, axis=axis)
    else:
        out += a_max.item()
    return out


class HMMBakis:
    """Left-to-Right HMM trained with count engineering."""

    def __init__(self, n_states=N_STATES, n_symbols=N_SYMBOLS):
        self.N  = n_states
        self.M  = n_symbols
        self.A  = None  # (N, N) transition matrix
        self.B  = None  # (N, M) emission matrix
        self.pi = None  # (N,)  initial distribution

    # ── Training ──────────────────────────────────────────────────

    def _segment(self, T: int) -> np.ndarray:
        """Linear segmentation: uniformly assign frames to states."""
        boundaries = np.linspace(0, T, self.N + 1).astype(int)
        seg = np.zeros(T, dtype=int)
        for i in range(self.N):
            seg[boundaries[i]:boundaries[i + 1]] = i
        return seg

    def train(self, sequences: list) -> None:
        """Train by count engineering (linear segmentation + frequency counts).

        Args:
            sequences: list of 1-D int32 VQ index arrays
        """
        B_counts = np.zeros((self.N, self.M))
        A_self   = np.zeros(self.N)
        A_next   = np.zeros(self.N - 1)

        for seq in sequences:
            T = len(seq)
            if T < self.N:
                continue
            seg = self._segment(T)
            for t, obs in enumerate(seq):
                B_counts[seg[t], int(obs)] += 1
            for i in range(self.N):
                cnt = int(np.sum(seg == i))
                if cnt > 0:
                    A_self[i] += max(cnt - 1, 0)
                    if i < self.N - 1:
                        A_next[i] += 1

        # Emission matrix with Laplace smoothing
        B_counts += 1e-6
        self.B = B_counts / B_counts.sum(axis=1, keepdims=True)

        # Transition matrix (Bakis: diagonal + superdiagonal only)
        self.A = np.zeros((self.N, self.N))
        for i in range(self.N):
            total = A_self[i] + (A_next[i] if i < self.N - 1 else 0)
            if total < 1e-10:
                self.A[i, i] = 0.85
                if i < self.N - 1:
                    self.A[i, i + 1] = 0.15
            else:
                self.A[i, i] = A_self[i] / total
                if i < self.N - 1:
                    self.A[i, i + 1] = A_next[i] / total
        # Final state is absorbing
        self.A[-1, :] = 0.0
        self.A[-1, -1] = 1.0

        self.pi = np.zeros(self.N)
        self.pi[0] = 1.0

    # ── Forward algorithm ─────────────────────────────────────────

    def log_likelihood(self, sequence):
        """Forward algorithm in log-space. Returns log P(O|λ)."""
        if self.A is None:
            return -np.inf
        T = len(sequence)
        if T == 0:
            return -np.inf

        log_A  = np.log(np.maximum(self.A,  1e-300))
        log_B  = np.log(np.maximum(self.B,  1e-300))
        log_pi = np.log(np.maximum(self.pi, 1e-300))

        log_alpha = log_pi + log_B[:, sequence[0]]
        for t in range(1, T):
            scores    = log_alpha[:, np.newaxis] + log_A
            log_alpha = _logsumexp(scores, axis=0) + log_B[:, sequence[t]]

        return float(_logsumexp(log_alpha))

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str) -> None:
        np.savez(path, A=self.A, B=self.B, pi=self.pi,
                 n_states=np.array(self.N), n_symbols=np.array(self.M))

    @classmethod
    def load(cls, path):
        data = np.load(path)
        hmm  = cls(int(data['n_states']), int(data['n_symbols']))
        hmm.A  = data['A']
        hmm.B  = data['B']
        hmm.pi = data['pi']
        return hmm
