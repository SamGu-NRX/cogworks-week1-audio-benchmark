"""Containment: ordinary broken submissions produce scored results, never crashes.

The threat model here is not a student gaming the benchmark. It is a student
writing broken code on a Tuesday. Every case below is a failure that either
appeared in an audited repository or is one line away from one, and the
assertion in each is the same: the run finishes, every case has an output,
the metrics are real numbers, and the student can read what happened.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from audio_identification_benchmark.contracts import Resources
from audio_identification_benchmark.datasets import (
    EnrollCase,
    QueryCase,
    load_manifest,
    materialize_cases,
)
from audio_identification_benchmark.drivers import fresh_copy, run_cases
from audio_identification_benchmark.metrics import score_outputs
from audio_identification_benchmark.plugins import AudioIdentificationBenchmark

from .fixtures.submissions import (
    BagOfHashesIdentifier,
    ChattyIdentifier,
    EnrollFailsIdentifier,
    MutatingIdentifier,
    NoneReturningIdentifier,
    RaisingIdentifier,
    RankedIdsIdentifier,
    ReferenceIdentifier,
    SingleAnswerIdentifier,
    TripleIdentifier,
    WrongShapeIdentifier,
)

SR = 44100


@pytest.fixture(scope="module")
def cases():
    """The shipped test tier: 8 songs, 68 queries. Rendered once."""

    return materialize_cases(load_manifest("test"))


@pytest.fixture
def resources(tmp_path):
    return Resources(sample_rate=SR, scratch_dir=tmp_path / "scratch")


@pytest.fixture(autouse=True)
def restore_cwd():
    """The driver chdirs on purpose; put the test process back afterwards."""

    original = os.getcwd()
    yield
    os.chdir(original)


def _run_and_score(factory, cases, resources):
    outputs = run_cases(factory, resources, cases)
    assert len(outputs) == len(cases), "every case must produce exactly one output"
    metrics, diagnostics = score_outputs(outputs, cases)
    for key, value in metrics.items():
        assert isinstance(value, float) and math.isfinite(value), (key, value)
    return outputs, metrics, diagnostics


# --------------------------------------------------------------------------
# The four containment cases named in the brief
# --------------------------------------------------------------------------


def test_a_raising_adapter_still_produces_a_scored_result(cases, resources):
    outputs, metrics, diagnostics = _run_and_score(RaisingIdentifier, cases, resources)
    assert metrics["identification_score"] == 0.0
    assert any("out of bounds" in line for line in diagnostics)
    # The enrollments succeeded; only identify raised. The report must say so
    # rather than blaming the database.
    assert all(output["ok"] for output in outputs if output["kind"] == "enroll")


def test_a_none_returning_adapter_scores_as_no_match(cases, resources):
    outputs, metrics, diagnostics = _run_and_score(NoneReturningIdentifier, cases, resources)
    assert metrics["identification_score"] == 0.0
    assert metrics["retrieval_failure_rate"] == 1.0
    query_outputs = [output for output in outputs if output["kind"] == "query"]
    assert all(output["ok"] for output in query_outputs)
    assert any("different hash spaces" in line for line in diagnostics)


def test_a_wrong_shape_adapter_is_refused_by_name_per_query(cases, resources):
    _, metrics, diagnostics = _run_and_score(WrongShapeIdentifier, cases, resources)
    assert metrics["identification_score"] == 0.0
    assert any("dict" in line and "song_id, score" in line for line in diagnostics)


def test_an_adapter_that_mutates_our_array_cannot_contaminate_later_queries(cases, resources):
    """The fixture zeroes what it is given during enroll and fills it with
    ones during identify. If the driver handed over the corpus itself, the
    second query would see an array of ones."""

    subset = _subset(cases, songs=3, queries=6)
    before = [np.array(case.samples) for case in subset if isinstance(case, EnrollCase)]
    _, metrics, _ = _run_and_score(MutatingIdentifier, subset, resources)
    after = [case.samples for case in subset if isinstance(case, EnrollCase)]
    for original, current in zip(before, after):
        assert original.tobytes() == current.tobytes(), "the corpus was mutated in place"
    assert math.isfinite(metrics["identification_score"])


# --------------------------------------------------------------------------
# Other real failures
# --------------------------------------------------------------------------


def test_a_failed_enrollment_marks_its_own_song_and_names_the_cause(cases, resources):
    subset = _subset(cases, songs=4, queries=12)
    outputs, metrics, diagnostics = _run_and_score(EnrollFailsIdentifier, subset, resources)
    failed = [
        output
        for output in outputs
        if output["kind"] == "enroll" and not output["ok"]
    ]
    assert len(failed) == 1
    assert "could not pickle" in failed[0]["error"]
    marked = [output for output in outputs if output.get("outcome") == "enrollment_failure"]
    assert marked, "queries for the failed song must carry the enrollment failure"
    assert "could not pickle" in marked[0]["error"]
    # The other songs still scored.
    assert metrics["identification_score"] > 0.0


def test_an_unmappable_submission_fails_every_case_with_the_mapping_report(cases, resources):
    class Mystery:
        def do_the_thing(self, stuff):
            return stuff

    outputs, metrics, diagnostics = _run_and_score(lambda r: Mystery(), cases, resources)
    assert all(not output["ok"] for output in outputs)
    assert metrics["identification_score"] == 0.0
    assert any("do_the_thing" in output["error"] for output in outputs)
    assert any("submission.py" in output["error"] for output in outputs)


def test_a_factory_that_raises_does_not_end_the_run(cases, resources):
    def factory(_resources):
        raise ImportError("No module named 'librosa'")

    outputs, metrics, _ = _run_and_score(factory, cases, resources)
    assert all(not output["ok"] for output in outputs)
    assert any("librosa" in output["error"] for output in outputs)


def test_a_chatty_submission_does_not_break_anything(cases, resources, capsys):
    subset = _subset(cases, songs=2, queries=4)
    _, metrics, _ = _run_and_score(ChattyIdentifier, subset, resources)
    assert math.isfinite(metrics["identification_score"])


def test_the_driver_chdirs_into_the_scratch_directory(cases, resources):
    """One audited repo writes a module-global relative db.pkl on every add
    and query; two runs in one directory corrupt each other."""

    seen = {}

    class WritesRelativePaths(ReferenceIdentifier):
        def enroll(self, song_id, samples, sample_rate):
            seen["cwd"] = Path(os.getcwd()).resolve()
            Path("db.pkl").write_bytes(b"database")
            super().enroll(song_id, samples, sample_rate)

    subset = _subset(cases, songs=1, queries=1)
    run_cases(WritesRelativePaths, resources, subset)
    assert seen["cwd"] == resources.scratch_dir.resolve()
    assert (resources.scratch_dir / "db.pkl").is_file()


def test_each_song_is_enrolled_exactly_once(cases, resources):
    identifier = ReferenceIdentifier()
    subset = _subset(cases, songs=4, queries=4)
    run_cases(lambda r: identifier, resources, subset)
    assert set(identifier.enroll_counts.values()) == {1}


def test_a_duplicate_enrollment_case_is_refused_rather_than_doubling_votes(resources):
    """Re-enrolling doubles a song's votes (measured 118 -> 236), which would
    silently advantage whichever song we sent twice."""

    identifier = ReferenceIdentifier()
    signal = np.zeros(SR, dtype=np.float32)
    duplicated = [
        EnrollCase("song-00", signal, SR),
        EnrollCase("song-00", signal, SR),
    ]
    outputs = run_cases(lambda r: identifier, resources, duplicated)
    assert outputs[0]["ok"] and not outputs[1]["ok"]
    assert "exactly once" in outputs[1]["error"]
    assert identifier.enroll_counts["song-00"] == 1


def test_a_warm_up_failure_is_a_note_not_a_run_failure(cases, resources):
    class WarmUpBreaks(ReferenceIdentifier):
        def __init__(self, resources=None):
            super().__init__(resources)
            self.calls = 0

        def identify(self, samples, sample_rate):
            self.calls += 1
            if self.calls == 1:
                raise ZeroDivisionError("float division by zero")
            return super().identify(samples, sample_rate)

    subset = _subset(cases, songs=2, queries=4)
    outputs, metrics, diagnostics = _run_and_score(WarmUpBreaks, subset, resources)
    assert any("warm-up" in line for line in diagnostics)
    assert all(output["ok"] for output in outputs if output["kind"] == "query")


# --------------------------------------------------------------------------
# Accepted return shapes, end to end
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "submission",
    [ReferenceIdentifier, RankedIdsIdentifier, SingleAnswerIdentifier, TripleIdentifier],
)
def test_every_accepted_shape_scores_the_same_clean_queries(cases, resources, submission):
    subset = _clean_only(cases, songs=4)
    _, metrics, _ = _run_and_score(submission, subset, resources)
    assert metrics["identification_score"] == 1.0


def test_ranked_ids_lose_only_the_margin_column(cases, resources):
    subset = _clean_only(cases, songs=4, include_out_of_set=True)
    _, with_scores, _ = _run_and_score(ReferenceIdentifier, subset, resources)
    _, without_scores, diagnostics = _run_and_score(RankedIdsIdentifier, subset, resources)
    assert with_scores["identification_score"] == without_scores["identification_score"]
    assert "margin_separation" in with_scores
    assert "margin_separation" not in without_scores
    assert any("not measured" in line for line in diagnostics)


# --------------------------------------------------------------------------
# Ordering, not magic numbers
# --------------------------------------------------------------------------


def test_tuned_beats_trivial_beats_chance(cases, resources):
    """An ordering with no gate, and a deliberately narrow claim.

    Absolute thresholds are absent because a number written without a
    calibration run behind it is a claim the code cannot support.

    What this test does NOT assert, and why: tuned above detuned. Calibration
    measured the primary metric on this grid at 0.5208 for tuned and 0.5333
    for a neighborhood of 51, which is 4.7x worse at real retrieval (0.200
    vs 0.933 top-1 on half-second clips). The grid does not rank, which is
    why the catalog row ships inactive. Asserting an ordering the instrument
    cannot deliver would either fail honestly or, worse, get loosened until
    it passed and become a permanently green test guarding nothing.

    The trivial baseline also measured at chance on this corpus rather than
    above it, so the assertion is that tuned clears the baseline, not that
    the baseline is a strong floor.
    """

    subset = _subset(cases, songs=6, queries=None)
    benchmark = AudioIdentificationBenchmark()
    baseline = benchmark.score(
        run_cases(ReferenceIdentifier, resources, subset), subset
    )["trivial_baseline_top1"]
    chance = 1.0 / 6

    _, tuned, _ = _run_and_score(ReferenceIdentifier, subset, resources)

    # The comparator must be a submission that actually ANSWERS on clean
    # audio, or the assertion below is vacuous. The previous comparator
    # (fanout=1, neighborhood=200, percentile=99.99) returns [] for every
    # clip, so `tuned > degraded` reduced to `x > 0.0` and was satisfied by
    # any submission that ever answered anything. neighborhood=51 is
    # genuinely bad and still answers: measured 0.200 vs 0.933 top-1 on
    # half-second clips.
    _, degraded, _ = _run_and_score(
        lambda r: ReferenceIdentifier(r, neighborhood=51),
        subset,
        resources,
    )
    assert degraded["retrieval_failure_rate"] < 1.0, (
        "the comparator must answer on clean audio, or the ordering "
        "assertion below cannot fail"
    )
    assert tuned["identification_score"] > baseline
    assert tuned["identification_score"] > chance
    assert baseline >= chance


def test_pitch_is_the_axis_that_separates_and_clean_is_not(cases, resources):
    """Reproduces the measured student result: clean saturates, pitch collapses.

    This is why the grid is built around pitch. If a future corpus change
    made clean accuracy fall or pitch accuracy saturate, the instrument
    would stop discriminating and this test is where that shows up.
    """

    subset = _subset(cases, songs=8, queries=None)
    _, metrics, _ = _run_and_score(ReferenceIdentifier, subset, resources)
    assert metrics["clean_top1"] == 1.0
    assert metrics["pitch_top1"] < metrics["clean_top1"]


def test_offset_alignment_does_not_separate_on_this_corpus(cases, resources):
    """The deleted probe, kept as a measurement so nobody rebuilds it.

    A bag-of-hashes matcher that never aligns time offsets scores close to
    the correct matcher, because (f1, f2, dt) hashes already encode local
    ordering. Two independent reviewers measured a maximum gap of 0.18
    against a proposed 0.50 gate, and at one block size the ablation won.
    The assertion is that the gap stays small; if it ever grows past 0.5
    the corpus has changed character and the finding needs re-deriving.
    """

    subset = _clean_only(cases, songs=6)
    _, correct, _ = _run_and_score(ReferenceIdentifier, subset, resources)
    _, ablated, _ = _run_and_score(BagOfHashesIdentifier, subset, resources)
    gap = correct["identification_score"] - ablated["identification_score"]
    assert gap < 0.5, gap


def test_a_sample_rate_mismatch_is_caught_and_named(cases, resources):
    """The silent wrong answer the taxonomy exists to name.

    A database built at 16 kHz and queried at 44.1 kHz puts the two hash
    spaces out of alignment, because fingerprint times are spectrogram
    column indices. The design note predicted this would present as ~100%
    retrieval failure; on this corpus it does not, and the difference is
    worth stating rather than asserting around.

    Measured on the test tier with six songs: gold's median vote count falls
    from 48.5 to 8.0, but gold still appears in the candidate list, because
    with only six songs a handful of surviving coincidental hash collisions
    is enough to keep it there. The outcome moves from 27 top-1 / 20 top-k /
    1 ranking-failure to 11 / 28 / 9. So the observable is the collapse in
    ranking, not the disappearance of the candidate, and the diagnostic has
    to name resampling on the ranking-failure branch too. The 100%
    retrieval-failure prediction presumably holds on a catalog large enough
    that coincidental collisions cannot carry gold; that has not been
    measured here and is not asserted.
    """

    class EnrollsAtADifferentRate(ReferenceIdentifier):
        def enroll(self, song_id, samples, sample_rate):
            count = int(len(samples) * 16000 / 44100)
            resampled = np.interp(
                np.linspace(0, len(samples) - 1, count),
                np.arange(len(samples)),
                samples.astype(np.float64),
            ).astype(np.float32)
            super().enroll(song_id, resampled, 16000)

    subset = _subset(cases, songs=6, queries=None)
    _, healthy, _ = _run_and_score(ReferenceIdentifier, subset, resources)
    _, broken, diagnostics = _run_and_score(EnrollsAtADifferentRate, subset, resources)
    assert broken["identification_score"] < healthy["identification_score"]
    assert broken["ranking_failure_rate"] > healthy["ranking_failure_rate"]
    assert any("resamples both" in line for line in diagnostics)


# --------------------------------------------------------------------------
# The plugin surface the controller calls
# --------------------------------------------------------------------------


def test_the_plugin_exposes_the_v2_interface():
    benchmark = AudioIdentificationBenchmark()
    assert benchmark.benchmark_id == "audio-identification"
    assert benchmark.benchmark_version == 1
    assert benchmark.contract_version == "cogworks.submissions.v2"
    assert benchmark.primary_metric == "identification_score"
    assert benchmark.primary_metric in benchmark.metric_labels
    assert benchmark.cache_status("test").ready
    resources = benchmark.model_factory()
    assert resources.scratch_dir is not None and resources.scratch_dir.is_dir()


def test_a_full_plugin_run_scores_and_labels_every_metric(cases, resources):
    benchmark = AudioIdentificationBenchmark()
    subset = _subset(cases, songs=4, queries=16)
    outputs = benchmark.run(ReferenceIdentifier, resources, subset)
    metrics = benchmark.score(outputs, subset)
    assert benchmark.primary_metric in metrics
    for key in metrics:
        assert key in benchmark.metric_labels, key
    assert benchmark.last_diagnostics
    import json

    json.dumps({"metrics": metrics, "outputs": outputs})


def test_fresh_copy_is_writeable_contiguous_float32():
    source = np.arange(8, dtype=np.float64)
    source.flags.writeable = False
    copy = fresh_copy(source)
    assert copy.dtype == np.float32
    assert copy.flags["C_CONTIGUOUS"] and copy.flags["WRITEABLE"]
    copy[0] = 99.0  # must not raise


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _subset(cases, songs, queries):
    """The first ``songs`` enrollments plus queries that only reference them."""

    kept = [case for case in cases if isinstance(case, EnrollCase)][:songs]
    ids = {case.song_id for case in kept}
    chosen = [
        case
        for case in cases
        if isinstance(case, QueryCase)
        and (case.gold_song_id in ids or case.kind == "out_of_set")
    ]
    if queries is not None:
        chosen = chosen[:queries]
    return kept + chosen


def _clean_only(cases, songs, include_out_of_set=False):
    """Clean, full-length, in-set queries: the cell every pipeline should win."""

    kept = [case for case in cases if isinstance(case, EnrollCase)][:songs]
    ids = {case.song_id for case in kept}
    chosen = [
        case
        for case in cases
        if isinstance(case, QueryCase)
        and case.gold_song_id in ids
        and not case.pitch_semitones
        and case.snr_db is None
        and case.clip_seconds >= 6.0
    ]
    if include_out_of_set:
        chosen += [case for case in cases if getattr(case, "kind", "") == "out_of_set"]
    return kept + chosen
