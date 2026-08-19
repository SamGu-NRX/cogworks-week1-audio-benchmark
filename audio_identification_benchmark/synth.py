"""The corpus generator: seeded, non-stationary, polyphonic synthetic music.

The course sets no evaluation corpus for Week 1 (students record their own
clips), so the benchmark has to bring one. Downloading real music is not an
option we can ship, and a corpus of sustained tones is worse than useless:
a stationary signal gives near-identical spectral peaks in every frame, the
offset histogram that a fingerprint matcher votes over degenerates, and the
instrument stops measuring the thing it exists to measure. So the generator
is built around non-stationarity, and it measures its own output rather than
asserting it (see ``corpus_stats``).

Each song is a seeded note-event process: a walk over a scale with chord
changes, three voices, sharp attacks (transients localize spectral peaks in
time), a per-song harmonic timbre, its own key and tempo, and a percussive
track of filtered noise bursts, because in real music the broadband
transients are what make peak-picking non-trivial.

Determinism is a hard requirement: the laptop render and the hosted render
must be the same bytes, or the sha256 pins in the manifest are theatre. Three
rules enforce it.

Every song draws from ``np.random.default_rng`` seeded from
``(master_seed, song_index)`` alone, never from global state.

All arithmetic runs in float64, with exactly one cast to float32 at the final
store.

And every transcendental, resampling, and interval-scaling call goes through
``exactmath`` rather than numpy. This is the one that is easy to get wrong.
Float64 pins how each operation *rounds*; it does not pin *which* operation
runs, and that is what actually varies: ``np.sin``, ``np.exp``, ``np.power``,
``np.convolve``, ``np.interp``, and ``Generator.uniform`` all returned
different bits under numpy 1.24 on Linux/x86 than under numpy 2.4 on
macOS/arm64, from identical inputs. The RNG bit stream itself was identical
throughout. See ``docs/decisions/week1-corpus-determinism.md`` for the
measurement and ``exactmath.py`` for the replacements.

Nothing in this module runs inside the scoring path except ``perturb`` and
the small spectrum helpers, and all of it is pure numpy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import exactmath

#: Version stamp for the rendering algorithm. Any change to the synthesis
#: math changes every sha256 in every manifest, so this moves with it and
#: the manifests are rebuilt rather than edited.
CORPUS_VERSION = "synth-v1"

#: Semitone offsets of the major and natural-minor scales.
_MAJOR = (0, 2, 4, 5, 7, 9, 11)
_MINOR = (0, 2, 3, 5, 7, 8, 10)

#: Triads by scale degree, as scale-degree offsets (root, third, fifth).
_TRIAD = (0, 2, 4)

#: log2(10). `10**k` is `exp2(k*log2(10))`; going through exp2 keeps the
#: perturbation deterministic for the same reason the renderer does.
_LOG2_10 = 3.321928094887362


class SynthError(RuntimeError):
    """The rendered corpus does not match what the manifest pinned."""


@dataclass(frozen=True)
class SongSpec:
    """Everything needed to render one song, and the hash it must produce."""

    song_id: str
    seed: int
    duration_seconds: float
    sha256: Optional[str] = None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _midi_to_hz(midi: np.ndarray) -> np.ndarray:
    return 440.0 * exactmath.power2((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


def _note(
    frequency: float,
    duration: float,
    sample_rate: int,
    partial_gains: np.ndarray,
    attack_seconds: float,
    decay_tau: float,
) -> np.ndarray:
    """One plucked/struck note: harmonic stack under a sharp AD envelope.

    The envelope matters more than the timbre here. A short attack puts a
    broadband transient at a known instant, which is what gives a
    spectrogram a peak that is localized in time as well as frequency; a
    slow swell would smear the same energy across many frames and make
    every frame look alike.
    """

    count = int(round(duration * sample_rate))
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    time = np.arange(count, dtype=np.float64) / float(sample_rate)

    # Harmonics, dropped once they pass Nyquist so nothing aliases back
    # down into the band the student's spectrogram will look at.
    harmonics = np.arange(1, partial_gains.shape[0] + 1, dtype=np.float64)
    frequencies = frequency * harmonics
    audible = frequencies < (0.5 * sample_rate) * 0.98
    if not audible.any():
        return np.zeros(count, dtype=np.float64)
    phase = 2.0 * np.pi * np.outer(frequencies[audible], time)
    wave = (partial_gains[audible, np.newaxis] * exactmath.sin(phase)).sum(axis=0)

    attack = max(int(round(attack_seconds * sample_rate)), 1)
    envelope = exactmath.exp(-time / decay_tau)
    if attack < count:
        envelope[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float64)
    else:
        envelope *= np.linspace(0.0, 1.0, count, dtype=np.float64)
    return wave * envelope


def _smooth(signal: np.ndarray, width: int) -> np.ndarray:
    """Box-average lowpass. Pure numpy stand-in for a filter design."""

    if width <= 1:
        return signal
    return exactmath.box_average(signal, width)


def _percussive_hit(
    kind: str, sample_rate: int, rng: np.random.Generator
) -> np.ndarray:
    """A noise burst shaped into a kick, a snare, or a hat.

    Real percussion is broadband and extremely short. This is what makes
    peak-picking non-trivial: a transient deposits energy in every frequency
    bin of one frame, so a naive threshold picks a whole column of peaks
    unless the student's neighborhood logic actually works.
    """

    if kind == "kick":
        duration, tau, width = 0.18, 0.045, 48
    elif kind == "snare":
        duration, tau, width = 0.13, 0.035, 6
    else:  # hat
        duration, tau, width = 0.06, 0.012, 1

    count = int(round(duration * sample_rate))
    time = np.arange(count, dtype=np.float64) / float(sample_rate)
    noise = rng.standard_normal(count)
    if kind == "kick":
        # Low thud: lowpassed noise plus a falling sine, the usual recipe.
        body = _smooth(noise, width) * 3.0
        sweep = exactmath.sin(2.0 * np.pi * (110.0 * exactmath.exp(-time / 0.03)) * time)
        shaped = 0.4 * body + 0.9 * sweep
    elif kind == "snare":
        shaped = _smooth(noise, width) * 1.6
    else:
        # Hat: subtract the lowpass to keep the high band only.
        shaped = noise - _smooth(noise, 9)
    return shaped * exactmath.exp(-time / tau)


def render_song(spec: SongSpec, sample_rate: int) -> np.ndarray:
    """Render one song to a float32 mono signal in ``[-1, 1]``.

    Deterministic from ``(spec.seed, spec.duration_seconds, sample_rate)``
    alone; no global RNG is read or written.
    """

    rng = np.random.default_rng(int(spec.seed))
    total = int(round(spec.duration_seconds * sample_rate))
    if total <= 0:
        raise SynthError("Song {} has non-positive duration.".format(spec.song_id))
    # float64 throughout; the single cast to float32 is at the return.
    mix = np.zeros(total, dtype=np.float64)

    tempo = float(rng.integers(84, 156))
    beat = 60.0 / tempo
    root = int(rng.integers(45, 60))
    scale = _MAJOR if rng.random() < 0.5 else _MINOR

    # Per-song timbre: how fast the harmonic series rolls off, plus a jitter
    # per partial. Two songs in the same key therefore do not share a
    # spectrum, which is what keeps cross-song hash overlap low.
    partial_count = int(rng.integers(5, 9))
    rolloff = float(exactmath.uniform(rng, 0.9, 2.1))
    gains = exactmath.power(np.arange(1, partial_count + 1, dtype=np.float64), -rolloff)
    gains = gains * exactmath.uniform(rng, 0.6, 1.4, size=partial_count)
    gains = gains / np.sum(gains)

    attack = float(exactmath.uniform(rng, 0.001, 0.006))
    # Decay is short relative to the beat on purpose: overlapping sustained
    # notes are what produce a stationary spectrum.
    decay_scale = float(exactmath.uniform(rng, 0.35, 0.85))

    # --- pitched voices -------------------------------------------------
    # A chord change every four beats; within a chord, the lead walks over
    # chord tones and passing tones, the mid arpeggiates, the bass holds
    # the root. Together that gives a spectrum that changes every eighth
    # note rather than every four bars.
    bars = int(np.ceil(spec.duration_seconds / (beat * 4.0))) + 1
    degree = 0
    for bar in range(bars):
        degree = int((degree + rng.integers(-3, 4)) % len(scale))
        chord = [scale[(degree + step) % len(scale)] + 12 * ((degree + step) // len(scale))
                 for step in _TRIAD]
        bar_start = bar * 4.0 * beat

        # Bass: root of the chord, one note per two beats.
        for half in range(2):
            start = bar_start + half * 2.0 * beat
            midi = root - 12 + chord[0]
            note = _note(
                float(_midi_to_hz(np.array([midi]))[0]),
                2.0 * beat * decay_scale,
                sample_rate,
                gains,
                attack,
                decay_scale * beat * 1.2,
            )
            _add_at(mix, note, int(round(start * sample_rate)), 0.55)

        # Mid: arpeggio on the beat.
        for step in range(4):
            start = bar_start + step * beat
            midi = root + chord[step % len(chord)]
            note = _note(
                float(_midi_to_hz(np.array([midi]))[0]),
                beat * decay_scale,
                sample_rate,
                gains,
                attack,
                decay_scale * beat * 0.7,
            )
            _add_at(mix, note, int(round(start * sample_rate)), 0.35)

        # Lead: eighth notes, chord tones with passing tones and rests.
        for step in range(8):
            if rng.random() < 0.25:
                continue
            start = bar_start + step * 0.5 * beat
            if rng.random() < 0.65:
                interval = chord[int(rng.integers(0, len(chord)))]
            else:
                interval = scale[int(rng.integers(0, len(scale)))]
            midi = root + 12 + interval + 12 * int(rng.integers(0, 2))
            note = _note(
                float(_midi_to_hz(np.array([midi]))[0]),
                0.5 * beat * decay_scale,
                sample_rate,
                gains,
                attack,
                decay_scale * beat * 0.35,
            )
            _add_at(mix, note, int(round(start * sample_rate)), 0.3 + 0.2 * rng.random())

    # --- percussion -----------------------------------------------------
    hits = int(np.ceil(spec.duration_seconds / (beat * 0.5)))
    for step in range(hits):
        start = int(round(step * 0.5 * beat * sample_rate))
        if step % 8 == 0 or step % 8 == 6:
            _add_at(mix, _percussive_hit("kick", sample_rate, rng), start, 0.9)
        if step % 8 == 4:
            _add_at(mix, _percussive_hit("snare", sample_rate, rng), start, 0.6)
        if rng.random() < 0.7:
            _add_at(mix, _percussive_hit("hat", sample_rate, rng), start, 0.18)

    peak = float(np.max(np.abs(mix)))
    if peak > 0.0:
        mix = mix * (0.89 / peak)
    # THE cast, and the only one. float32 arithmetic would be a second source
    # of cross-machine difference on top of the libm one exactmath handles:
    # a float32 oscillator picks a different vectorized kernel again. Doing
    # everything in float64 and narrowing once at the end keeps the corpus one
    # deterministic computation with one rounding step.
    return np.ascontiguousarray(mix, dtype=np.float32)


def _add_at(mix: np.ndarray, block: np.ndarray, start: int, gain: float) -> None:
    """Mix ``block`` into ``mix`` at ``start``, truncated at the end."""

    if start >= mix.shape[0] or block.shape[0] == 0:
        return
    if start < 0:
        block = block[-start:]
        start = 0
    end = min(start + block.shape[0], mix.shape[0])
    mix[start:end] += gain * block[: end - start]


def sha256_signal(signal: np.ndarray) -> str:
    """Hash the exact float32 bytes that will be handed to a submission."""

    array = np.ascontiguousarray(signal, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def render_corpus(
    specs: Sequence[SongSpec], sample_rate: int, verify: bool = True
) -> Dict[str, np.ndarray]:
    """Render every song, verifying each against its pinned sha256.

    A mismatch is a platform failure, raised before any student code runs.
    Scoring a submission against a corpus that is not the pinned one would
    produce a number that means nothing and looks like a number that does.
    """

    signals: Dict[str, np.ndarray] = {}
    for spec in specs:
        signal = render_song(spec, sample_rate)
        if verify and spec.sha256:
            digest = sha256_signal(signal)
            if digest != spec.sha256:
                raise SynthError(
                    "Rendered {} does not match the manifest: got {}, expected {}. The "
                    "corpus generator, numpy, or the platform changed; do not score "
                    "against this audio.".format(spec.song_id, digest[:16], spec.sha256[:16])
                )
        signals[spec.song_id] = signal
    return signals


# --------------------------------------------------------------------------
# Perturbations (pure numpy; these run inside the scoring path)
# --------------------------------------------------------------------------


def pitch_shift(signal: np.ndarray, semitones: float, length: Optional[int] = None) -> np.ndarray:
    """Resample-based pitch shift, then trim or pad back to ``length``.

    Shifting up by ``n`` semitones means reading the signal at ``2**(n/12)``
    samples per output sample, which raises every frequency by that factor
    and shortens the result; the trim/pad restores the requested duration.
    Linear interpolation is deliberate: it keeps the scoring path free of
    scipy, and the interpolation artifacts land far below the spectral
    peaks a fingerprinter picks.
    """

    source = np.asarray(signal, dtype=np.float64)
    target = int(source.shape[0] if length is None else length)
    if source.shape[0] == 0 or target <= 0:
        return np.zeros(max(target, 0), dtype=np.float32)
    if semitones == 0:
        shifted = source
    else:
        rate = float(exactmath.power2(np.float64(semitones) / 12.0))
        count = int(np.floor((source.shape[0] - 1) / rate)) + 1
        positions = np.arange(count, dtype=np.float64) * rate
        shifted = exactmath.interp(
            positions, np.arange(source.shape[0], dtype=np.float64), source
        )
    if shifted.shape[0] >= target:
        out = shifted[:target]
    else:
        out = np.concatenate([shifted, np.zeros(target - shifted.shape[0], dtype=np.float64)])
    return np.ascontiguousarray(out, dtype=np.float32)


def add_noise_at_snr(
    signal: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Additive white Gaussian noise at a target SNR.

    The formula is the one the audited student harness used, kept identical
    so their committed sweeps stay comparable to ours::

        noise_power = signal_power / 10 ** (snr_db / 10)
    """

    source = np.asarray(signal, dtype=np.float64)
    if source.shape[0] == 0:
        return np.ascontiguousarray(source, dtype=np.float32)
    signal_power = float(np.mean(source ** 2))
    if signal_power <= 0.0:
        return np.ascontiguousarray(source, dtype=np.float32)
    noise_power = signal_power / float(exactmath.exp2(np.float64(snr_db) / 10.0 * _LOG2_10))
    noise = rng.standard_normal(source.shape[0]) * np.sqrt(noise_power)
    mixed = source + noise
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = mixed / peak
    return np.ascontiguousarray(mixed, dtype=np.float32)


def extract_clip(
    signal: np.ndarray, sample_rate: int, seconds: float, offset_seconds: float
) -> np.ndarray:
    """A clip of ``seconds`` starting at ``offset_seconds``, zero-padded."""

    count = int(round(seconds * sample_rate))
    start = int(round(offset_seconds * sample_rate))
    start = max(min(start, max(signal.shape[0] - 1, 0)), 0)
    piece = np.asarray(signal[start : start + count], dtype=np.float32)
    if piece.shape[0] < count:
        piece = np.concatenate(
            [piece, np.zeros(count - piece.shape[0], dtype=np.float32)]
        )
    return np.ascontiguousarray(piece, dtype=np.float32)


def perturb(
    signal: np.ndarray,
    sample_rate: int,
    seconds: float,
    offset_seconds: float,
    pitch_semitones: float = 0.0,
    snr_db: Optional[float] = None,
    seed: int = 0,
) -> np.ndarray:
    """Clip, pitch-shift, and add noise, in that order, deterministically.

    Order matters and is fixed here: shifting a clip is cheaper than
    shifting a whole song, and noise added last is not itself pitch-shifted,
    which is what a microphone in a noisy room actually gives you.
    """

    clip = extract_clip(signal, sample_rate, seconds, offset_seconds)
    if pitch_semitones:
        clip = pitch_shift(clip, pitch_semitones, clip.shape[0])
    if snr_db is not None:
        clip = add_noise_at_snr(clip, snr_db, np.random.default_rng(int(seed)))
    return np.ascontiguousarray(clip, dtype=np.float32)


# --------------------------------------------------------------------------
# Spectra: used by corpus statistics and by the trivial baseline
# --------------------------------------------------------------------------

#: FFT size and hop used for every spectrum in this package. Chosen to match
#: the course's own recommended ``mlab.specgram(NFFT=4096, noverlap=2048)``
#: so corpus statistics describe the representation students actually build.
NFFT = 4096
HOP = 2048
_LOG_FLOOR = 1e-20


def log_spectrogram(
    signal: np.ndarray, nfft: int = NFFT, hop: int = HOP
) -> np.ndarray:
    """``(frames, bins)`` log-magnitude spectrogram. Pure numpy."""

    source = np.asarray(signal, dtype=np.float64)
    if source.shape[0] < nfft:
        source = np.concatenate([source, np.zeros(nfft - source.shape[0], dtype=np.float64)])
    frames = 1 + (source.shape[0] - nfft) // hop
    if frames <= 0:
        return np.zeros((0, nfft // 2 + 1), dtype=np.float64)
    window = np.hanning(nfft)
    indices = np.arange(nfft)[np.newaxis, :] + hop * np.arange(frames)[:, np.newaxis]
    blocks = source[indices] * window[np.newaxis, :]
    magnitude = np.abs(np.fft.rfft(blocks, axis=1))
    return np.log(np.clip(magnitude, _LOG_FLOOR, None))


def mean_log_spectrum(signal: np.ndarray, nfft: int = NFFT, hop: int = HOP) -> np.ndarray:
    """One vector per clip: the whole-clip average log spectrum.

    This is the feature behind ``trivial_baseline_top1``. It throws away all
    temporal structure, which is exactly the point: it is the floor a
    submission has to beat before "it works" means anything.
    """

    spectrogram = log_spectrogram(signal, nfft, hop)
    if spectrogram.shape[0] == 0:
        return np.zeros(nfft // 2 + 1, dtype=np.float64)
    return spectrogram.mean(axis=0)


def spectral_flux(signal: np.ndarray, nfft: int = NFFT, hop: int = HOP) -> float:
    """Mean frame-to-frame change in the log spectrum, per bin.

    The non-stationarity statistic. A sustained chord scores near zero (each
    frame looks like the last, so peak positions repeat and the offset
    histogram a matcher votes over degenerates); dense note events with
    sharp attacks score high. The generator reports this rather than
    claiming it.
    """

    spectrogram = log_spectrogram(signal, nfft, hop)
    if spectrogram.shape[0] < 2:
        return 0.0
    difference = np.diff(spectrogram, axis=0)
    return float(np.mean(np.abs(difference)))


# --------------------------------------------------------------------------
# Corpus statistics: a reference fingerprinter used ONLY for measurement
# --------------------------------------------------------------------------
#
# Nothing below is on the scoring path. It exists so the manifest can record
# what kind of corpus it pinned: how non-stationary the songs are and how
# much fingerprint vocabulary two different songs accidentally share. A
# corpus with high cross-song overlap would make top-1 accuracy a measure of
# our generator rather than of the submission.


def _rolling_max(matrix: np.ndarray, size: int, axis: int) -> np.ndarray:
    """Exact rectangular dilation along one axis (separable, so 2-D is two)."""

    if size <= 1:
        return matrix
    pad = size // 2
    widths = [(0, 0), (0, 0)]
    widths[axis] = (pad, size - 1 - pad)
    padded = np.pad(matrix, widths, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size, axis=axis)
    return windows.max(axis=-1)


def reference_peaks(
    spectrogram: np.ndarray, neighborhood: int = 15, percentile: float = 75.0
) -> np.ndarray:
    """Local maxima above the corpus's own amplitude threshold.

    Follows the course's stated defaults (a roughly 15-bin neighborhood and
    the 75th-percentile amplitude as the foreground threshold) so the
    statistics describe peaks a student would plausibly extract.
    """

    if spectrogram.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    dilated = _rolling_max(_rolling_max(spectrogram, neighborhood, 0), neighborhood, 1)
    threshold = float(np.percentile(spectrogram, percentile))
    mask = (spectrogram >= dilated) & (spectrogram > threshold)
    rows, columns = np.nonzero(mask)
    return np.stack([rows, columns], axis=1)


def reference_hashes(
    signal: np.ndarray, fanout: int = 15, max_peaks: int = 6000
) -> set:
    """``(f1, f2, dt)`` fanout hashes, the shape the capstone describes."""

    peaks = reference_peaks(log_spectrogram(signal))
    if peaks.shape[0] == 0:
        return set()
    order = np.lexsort((peaks[:, 1], peaks[:, 0]))  # by frame, then by bin
    peaks = peaks[order][:max_peaks]
    keys = set()
    frames = peaks[:, 0]
    bins = peaks[:, 1]
    for index in range(peaks.shape[0]):
        stop = min(index + 1 + fanout, peaks.shape[0])
        for other in range(index + 1, stop):
            keys.add((int(bins[index]), int(bins[other]), int(frames[other] - frames[index])))
    return keys


def corpus_stats(signals: Dict[str, np.ndarray], hash_songs: int = 6) -> Dict[str, float]:
    """Measured properties of a rendered corpus, recorded in the manifest.

    ``mean_spectral_flux`` is the non-stationarity statistic;
    ``mean_cross_song_hash_overlap`` is the Jaccard overlap of the reference
    fingerprint vocabularies of two different songs, averaged over pairs
    (real student music measured 1.5%, so anything near that is fine and a
    large number would mean the generator is making one song in N keys).
    """

    ids = sorted(signals)
    flux = [spectral_flux(signals[song_id]) for song_id in ids]
    subset = ids[: max(hash_songs, 2)]
    vocabularies = [reference_hashes(signals[song_id]) for song_id in subset]
    overlaps: List[float] = []
    for left in range(len(vocabularies)):
        for right in range(left + 1, len(vocabularies)):
            union = vocabularies[left] | vocabularies[right]
            if not union:
                continue
            overlaps.append(len(vocabularies[left] & vocabularies[right]) / float(len(union)))
    sizes = [len(vocabulary) for vocabulary in vocabularies]
    return {
        "mean_spectral_flux": float(np.mean(flux)) if flux else 0.0,
        "min_spectral_flux": float(np.min(flux)) if flux else 0.0,
        "mean_cross_song_hash_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "max_cross_song_hash_overlap": float(np.max(overlaps)) if overlaps else 0.0,
        "mean_reference_hashes_per_song": float(np.mean(sizes)) if sizes else 0.0,
        "hash_sample_songs": float(len(subset)),
    }


def specs_from_manifest(entries: Sequence[dict]) -> List[SongSpec]:
    """Manifest song rows to ``SongSpec`` objects."""

    return [
        SongSpec(
            song_id=str(entry["song_id"]),
            seed=int(entry["seed"]),
            duration_seconds=float(entry["duration_seconds"]),
            sha256=str(entry["sha256"]) if entry.get("sha256") else None,
        )
        for entry in entries
    ]


def song_seed(master_seed: int, index: int) -> int:
    """Deterministic per-song seed from ``(master_seed, index)`` alone.

    Spelled out rather than left to ``default_rng([master, index])`` so the
    manifest can record the integer and a reader can reproduce one song
    without reconstructing the generator's state.
    """

    digest = hashlib.sha256("{}:{}".format(int(master_seed), int(index)).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def render_from_manifest(
    manifest: dict, verify: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Render the enrolled catalog and the out-of-set songs from a manifest."""

    sample_rate = int(manifest["sample_rate"])
    catalog = render_corpus(specs_from_manifest(manifest["songs"]), sample_rate, verify)
    out_of_set = render_corpus(
        specs_from_manifest(manifest.get("out_of_set_songs", [])), sample_rate, verify
    )
    return catalog, out_of_set
