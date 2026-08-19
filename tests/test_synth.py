"""The corpus must be the same audio everywhere, and must not be stationary."""

from __future__ import annotations

import numpy as np
import pytest

from audio_identification_benchmark import synth
from audio_identification_benchmark.datasets import (
    assert_disjoint,
    load_manifest,
    materialize_cases,
)

SR = 44100


@pytest.fixture(scope="module")
def songs():
    return {
        "song-{:02d}".format(index): synth.render_song(
            synth.SongSpec("song-{:02d}".format(index), synth.song_seed(4242, index), 8.0), SR
        )
        for index in range(4)
    }


def test_render_is_byte_identical_for_the_same_seed():
    spec = synth.SongSpec("song-00", 12345, 6.0)
    first = synth.render_song(spec, SR)
    second = synth.render_song(spec, SR)
    assert first.dtype == np.float32
    assert first.tobytes() == second.tobytes()
    assert synth.sha256_signal(first) == synth.sha256_signal(second)


def test_render_ignores_global_rng_state():
    """A global seed set by student code (or by pytest) must not change audio."""

    spec = synth.SongSpec("song-00", 999, 4.0)
    np.random.seed(1)
    first = synth.render_song(spec, SR)
    np.random.seed(2)
    _ = np.random.random(100)
    second = synth.render_song(spec, SR)
    assert first.tobytes() == second.tobytes()


def test_different_seeds_give_different_songs():
    left = synth.render_song(synth.SongSpec("a", 1, 4.0), SR)
    right = synth.render_song(synth.SongSpec("b", 2, 4.0), SR)
    assert synth.sha256_signal(left) != synth.sha256_signal(right)


def test_signal_is_bounded_and_contiguous(songs):
    for signal in songs.values():
        assert signal.dtype == np.float32
        assert signal.flags["C_CONTIGUOUS"]
        assert float(np.max(np.abs(signal))) <= 1.0
        assert float(np.max(np.abs(signal))) > 0.5  # not silence


def test_sha256_mismatch_is_a_loud_failure():
    spec = synth.SongSpec("song-00", 7, 2.0, sha256="0" * 64)
    with pytest.raises(synth.SynthError) as info:
        synth.render_corpus([spec], SR)
    assert "does not match the manifest" in str(info.value)


def test_corpus_is_non_stationary(songs):
    """A sustained-chord corpus would destroy the offset histogram.

    The 1.0 floor is a sanity bound, not a calibrated threshold: a single
    held chord measures near 0.1 on this statistic and the generated songs
    measure around 4-7. It exists to catch a regression that removes the
    note-event process, not to encode a tuned value.
    """

    flux = [synth.spectral_flux(signal) for signal in songs.values()]
    assert min(flux) > 1.0, flux
    sustained = np.zeros(SR * 4, dtype=np.float32)
    time = np.arange(sustained.shape[0], dtype=np.float64) / SR
    for frequency in (220.0, 277.0, 330.0):
        sustained += (0.3 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    assert synth.spectral_flux(sustained) < min(flux)


def test_cross_song_hash_overlap_is_small(songs):
    """Two songs must not share a fingerprint vocabulary.

    High overlap would mean top-1 accuracy measures our generator rather
    than the submission. Real student music measured 1.5% and a synthetic
    corpus 3.6%; 0.20 is a loose ceiling that only a badly degenerate
    generator would cross.
    """

    stats = synth.corpus_stats(songs, hash_songs=4)
    assert stats["mean_cross_song_hash_overlap"] < 0.20, stats
    assert stats["mean_reference_hashes_per_song"] > 100, stats


def test_pitch_shift_moves_the_spectrum_and_keeps_the_length():
    signal = synth.render_song(synth.SongSpec("a", 31, 4.0), SR)
    shifted = synth.pitch_shift(signal, 2.0, signal.shape[0])
    assert shifted.shape == signal.shape
    assert shifted.dtype == np.float32
    original = synth.mean_log_spectrum(signal)
    moved = synth.mean_log_spectrum(shifted)
    # The spectral centroid must rise: two semitones is a factor of 1.12.
    bins = np.arange(original.shape[0], dtype=np.float64)
    weight_original = np.exp(original)
    weight_moved = np.exp(moved)
    centroid_original = float((bins * weight_original).sum() / weight_original.sum())
    centroid_moved = float((bins * weight_moved).sum() / weight_moved.sum())
    assert centroid_moved > centroid_original


def test_zero_semitone_shift_is_the_identity():
    signal = synth.render_song(synth.SongSpec("a", 31, 2.0), SR)
    assert synth.pitch_shift(signal, 0.0).tobytes() == signal.tobytes()


def test_noise_hits_the_requested_snr():
    signal = synth.render_song(synth.SongSpec("a", 77, 3.0), SR)
    for target in (20.0, 10.0, 0.0):
        noisy = synth.add_noise_at_snr(signal, target, np.random.default_rng(5))
        noise = noisy.astype(np.float64) - signal.astype(np.float64)
        measured = 10.0 * np.log10(np.mean(signal.astype(np.float64) ** 2) / np.mean(noise ** 2))
        assert abs(measured - target) < 1.0, (target, measured)


def test_perturb_is_deterministic():
    signal = synth.render_song(synth.SongSpec("a", 8, 6.0), SR)
    kwargs = dict(seconds=2.0, offset_seconds=1.0, pitch_semitones=1.0, snr_db=5.0, seed=11)
    first = synth.perturb(signal, SR, **kwargs)
    second = synth.perturb(signal, SR, **kwargs)
    assert first.tobytes() == second.tobytes()
    assert first.dtype == np.float32


def test_shipped_manifests_verify_and_are_disjoint():
    test = load_manifest("test")
    evaluation = load_manifest("evaluation")
    assert_disjoint([test, evaluation])
    assert test["corpus_version"] == synth.CORPUS_VERSION
    assert evaluation["corpus_version"] == synth.CORPUS_VERSION
    # Recorded so a regression in the generator is visible in review.
    assert test["corpus_stats"]["mean_spectral_flux"] > 1.0
    assert test["corpus_stats"]["mean_cross_song_hash_overlap"] < 0.20


def test_test_tier_materializes_with_hash_verification():
    manifest = load_manifest("test")
    cases = materialize_cases(manifest)
    enrolls = [case for case in cases if case.kind == "enroll"]
    queries = [case for case in cases if case.kind != "enroll"]
    assert len(enrolls) == len(manifest["songs"])
    assert len(queries) == len(manifest["queries"])
    assert {case.song_id for case in enrolls} == {row["song_id"] for row in manifest["songs"]}
    clip = queries[0].samples
    assert clip.dtype == np.float32
    assert clip.flags["C_CONTIGUOUS"]
    assert clip.shape[0] == int(round(queries[0].clip_seconds * queries[0].sample_rate))


def test_out_of_set_queries_have_no_gold():
    cases = materialize_cases(load_manifest("test"))
    unseen = [case for case in cases if getattr(case, "kind", "") == "out_of_set"]
    assert unseen
    assert all(case.gold_song_id is None for case in unseen)
    catalog = {case.song_id for case in cases if getattr(case, "kind", "") == "enroll"}
    assert all(case.source_song_id not in catalog for case in unseen)
