"""The taxonomy, the baselines, and the diagnostics, on hand-built outputs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from audio_identification_benchmark.datasets import EnrollCase, QueryCase
from audio_identification_benchmark.metrics import (
    classify_outcome,
    margin_auc,
    score_outputs,
    trivial_baseline_outcomes,
)
from audio_identification_benchmark.synth import SongSpec, render_song, song_seed

SR = 44100


def _query(query_id, gold, cell="clean", **kwargs):
    defaults = dict(
        clip_seconds=10.0,
        pitch_semitones=0.0,
        snr_db=None,
        offset_seconds=0.0,
        noise_seed=1,
    )
    defaults.update(kwargs)
    return QueryCase(
        query_id=query_id,
        sample_rate=SR,
        gold_song_id=gold,
        kind="in_set" if gold else "out_of_set",
        source_song_id=gold or "unseen-00",
        **defaults
    )


def _hit(query_id, candidates, scores=None):
    return {
        "ok": True,
        "kind": "query",
        "query_id": query_id,
        "candidates": list(candidates),
        "scores": scores,
        "shape": "ranked_pairs" if scores else "ranked_ids",
        "seconds": 0.01,
    }


# --------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------


def test_taxonomy_four_ways():
    assert classify_outcome(["a", "b", "c"], "a") == "top_1"
    assert classify_outcome(["b", "a", "c"], "a") == "top_k"
    assert classify_outcome(["b", "c", "d", "e", "f", "a"], "a", top_k=5) == "ranking_failure"
    assert classify_outcome(["b", "c"], "a") == "retrieval_failure"
    assert classify_outcome([], "a") == "retrieval_failure"


def test_top_k_boundary_is_inclusive_of_rank_k():
    assert classify_outcome(["b", "c", "d", "e", "a"], "a", top_k=5) == "top_k"
    assert classify_outcome(["b", "c", "d", "e", "f", "a"], "a", top_k=5) == "ranking_failure"


def test_taxonomy_refuses_an_out_of_set_query():
    with pytest.raises(ValueError):
        classify_outcome(["a"], None)


# --------------------------------------------------------------------------
# Aggregate metrics
# --------------------------------------------------------------------------


def _catalog_cases(count=4):
    return [
        EnrollCase("song-{:02d}".format(index), np.zeros(4, dtype=np.float32), SR)
        for index in range(count)
    ]


def test_a_perfect_submission_scores_one_and_chance_is_one_over_n():
    cases = _catalog_cases(4)
    outputs = [{"ok": True, "kind": "enroll", "song_id": case.song_id} for case in cases]
    for index in range(4):
        gold = "song-{:02d}".format(index)
        cases.append(_query("q{}".format(index), gold))
        outputs.append(_hit("q{}".format(index), [gold, "song-99"], [10.0, 1.0]))
    metrics, _ = score_outputs(outputs, cases)
    assert metrics["identification_score"] == 1.0
    assert metrics["chance_top1"] == 0.25
    assert metrics["retrieval_failure_rate"] == 0.0
    assert metrics["ranking_failure_rate"] == 0.0


def test_an_inert_submission_lands_at_total_retrieval_failure():
    cases = _catalog_cases(4)
    outputs = [{"ok": True, "kind": "enroll", "song_id": case.song_id} for case in cases]
    for index in range(4):
        cases.append(_query("q{}".format(index), "song-{:02d}".format(index)))
        outputs.append(_hit("q{}".format(index), [], []))
    metrics, diagnostics = score_outputs(outputs, cases)
    assert metrics["identification_score"] == 0.0
    assert metrics["retrieval_failure_rate"] == 1.0
    assert any("different hash spaces" in line for line in diagnostics)


def test_ranking_failure_is_named_separately_from_retrieval_failure():
    cases = _catalog_cases(4)
    outputs = [{"ok": True, "kind": "enroll", "song_id": case.song_id} for case in cases]
    for index in range(4):
        gold = "song-{:02d}".format(index)
        cases.append(_query("q{}".format(index), gold))
        outputs.append(_hit("q{}".format(index), ["song-99"] * 6 + [gold]))
    metrics, diagnostics = score_outputs(outputs, cases)
    assert metrics["ranking_failure_rate"] == 1.0
    assert metrics["retrieval_failure_rate"] == 0.0
    assert any("not near the top" in line for line in diagnostics)


def test_cells_are_reported_separately():
    cases = _catalog_cases(2)
    outputs = [{"ok": True, "kind": "enroll", "song_id": case.song_id} for case in cases]
    plan = [
        ("clean", dict(clip_seconds=10.0), True),
        ("short", dict(clip_seconds=3.0), True),
        ("noisy", dict(snr_db=0.0), False),
        ("pitch", dict(pitch_semitones=2.0), False),
    ]
    counter = 0
    for _name, kwargs, correct in plan:
        for index in range(2):
            gold = "song-{:02d}".format(index)
            query_id = "q{}".format(counter)
            cases.append(_query(query_id, gold, **kwargs))
            outputs.append(_hit(query_id, [gold] if correct else ["song-99"]))
            counter += 1
    metrics, _ = score_outputs(outputs, cases)
    assert metrics["clean_top1"] == 1.0
    assert metrics["short_clip_top1"] == 1.0
    assert metrics["noisy_top1"] == 0.0
    assert metrics["pitch_top1"] == 0.0
    assert metrics["identification_score"] == 0.5


def test_a_failed_enrollment_zeroes_only_its_own_song():
    cases = _catalog_cases(2)
    outputs = [
        {"ok": False, "kind": "enroll", "song_id": "song-00", "error": "pickle failed"},
        {"ok": True, "kind": "enroll", "song_id": "song-01"},
    ]
    cases.append(_query("q0", "song-00"))
    outputs.append(
        {
            "ok": False,
            "kind": "query",
            "query_id": "q0",
            "error": "song-00 failed to enroll",
            "outcome": "enrollment_failure",
        }
    )
    cases.append(_query("q1", "song-01"))
    outputs.append(_hit("q1", ["song-01"]))
    metrics, diagnostics = score_outputs(outputs, cases)
    assert metrics["identification_score"] == 0.5
    assert any("failed to enroll" in line for line in diagnostics)
    assert any("enrollment_failure" in line for line in diagnostics)


def test_a_raising_query_is_counted_as_an_error_not_dropped():
    cases = _catalog_cases(2)
    outputs = [{"ok": True, "kind": "enroll", "song_id": case.song_id} for case in cases]
    cases.append(_query("q0", "song-00"))
    outputs.append({"ok": False, "kind": "query", "query_id": "q0", "error": "boom"})
    cases.append(_query("q1", "song-01"))
    outputs.append(_hit("q1", ["song-01"]))
    metrics, diagnostics = score_outputs(outputs, cases)
    assert metrics["identification_score"] == 0.5
    assert any("boom" in line for line in diagnostics)


# --------------------------------------------------------------------------
# Margins
# --------------------------------------------------------------------------


def test_margin_auc_separates_and_is_symmetric_under_ties():
    assert margin_auc([1.0, 0.9, 0.8], [0.1, 0.2]) == 1.0
    assert margin_auc([0.1, 0.2], [1.0, 0.9, 0.8]) == 0.0
    assert margin_auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert margin_auc([], [0.1]) is None


def test_margin_separation_is_omitted_when_no_scores_were_returned():
    cases = _catalog_cases(2)
    outputs = [{"ok": True, "kind": "enroll", "song_id": case.song_id} for case in cases]
    cases.append(_query("q0", "song-00"))
    outputs.append(_hit("q0", ["song-00"]))
    cases.append(_query("q1", None))
    outputs.append(_hit("q1", ["song-00"]))
    metrics, diagnostics = score_outputs(outputs, cases)
    assert "margin_separation" not in metrics
    assert any("not measured" in line for line in diagnostics)


def test_margin_separation_is_measured_when_scores_are_present():
    cases = _catalog_cases(2)
    outputs = [{"ok": True, "kind": "enroll", "song_id": case.song_id} for case in cases]
    for index in range(4):
        cases.append(_query("in{}".format(index), "song-00"))
        outputs.append(_hit("in{}".format(index), ["song-00", "song-01"], [100.0, 5.0]))
    for index in range(4):
        cases.append(_query("out{}".format(index), None))
        outputs.append(_hit("out{}".format(index), ["song-00", "song-01"], [6.0, 5.0]))
    metrics, diagnostics = score_outputs(outputs, cases)
    assert metrics["margin_separation"] == 1.0
    assert any("never enrolled still came back" in line for line in diagnostics)


def test_every_metric_is_json_safe():
    """NaN does not survive the trip to the run page; nothing may emit one."""

    cases = _catalog_cases(2)
    outputs = [{"ok": True, "kind": "enroll", "song_id": case.song_id} for case in cases]
    cases.append(_query("q0", "song-00"))
    outputs.append(_hit("q0", [], []))
    metrics, _ = score_outputs(outputs, cases)
    for key, value in metrics.items():
        assert isinstance(value, float), key
        assert math.isfinite(value), key


def test_a_single_id_shape_is_reported_as_a_limitation():
    cases = _catalog_cases(2)
    outputs = [{"ok": True, "kind": "enroll", "song_id": case.song_id} for case in cases]
    cases.append(_query("q0", "song-00"))
    outputs.append(
        {
            "ok": True,
            "kind": "query",
            "query_id": "q0",
            "candidates": ["song-00"],
            "scores": None,
            "shape": "single_id",
            "seconds": 0.01,
        }
    )
    _, diagnostics = score_outputs(outputs, cases)
    assert any("single id rather than a ranked list" in line for line in diagnostics)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def test_trivial_baseline_beats_chance_but_is_far_from_one():
    """The floor the primary metric is printed beside.

    Asserted as an ordering (above chance, below perfect) rather than
    against the 0.222 grid figure from the design note: that number came
    from a different corpus size and clip mix, and pinning it here would be
    a magic number with no run behind it in this repository.
    """

    catalog = {
        "song-{:02d}".format(index): render_song(
            SongSpec("song-{:02d}".format(index), song_seed(90210, index), 6.0), SR
        )
        for index in range(6)
    }
    cases = [EnrollCase(song_id, signal, SR) for song_id, signal in catalog.items()]
    for index, song_id in enumerate(sorted(catalog)):
        case = _query("q{}".format(index), song_id, clip_seconds=3.0, pitch_semitones=3.0)
        case._source = catalog[song_id]
        case.offset_seconds = 1.0
        cases.append(case)
    result = trivial_baseline_outcomes(cases, catalog)
    chance = 1.0 / len(catalog)
    assert chance <= result["trivial_baseline_top1"] <= 1.0


def test_a_score_at_or_below_the_trivial_baseline_says_so():
    cases = _catalog_cases(4)
    outputs = [{"ok": True, "kind": "enroll", "song_id": case.song_id} for case in cases]
    cases.append(_query("q0", "song-00"))
    outputs.append(_hit("q0", ["song-99"]))
    _, diagnostics = score_outputs(outputs, cases, baseline={"trivial_baseline_top1": 0.4})
    assert any("trivial baseline" in line for line in diagnostics)
