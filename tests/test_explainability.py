"""Every reported metric is explained, and the explanation names the course.

A benchmark that reports a number a student cannot trace back to something
they were taught is a black box, and a black box teaches nothing. These tests
make that a build-time property rather than an intention: a new metric with no
explanation, or an explanation written in benchmark jargon instead of the
course's own vocabulary, fails here.

The correspondence to the course is not decorative. Three of the four scored
axes -- clip length, noise level, catalog size -- are the ones the capstone
itself asks students to vary, and two more metrics answer questions the
capstone asks in prose (how large is the leading tally against the
next-largest, and does a clip from a song outside the database still return a
match). See `docs/cogweb/pages/Audio/capstone_summary.md`, "Analyzing and
Testing Performance".
"""

from __future__ import annotations

import re

import pytest

from audio_identification_benchmark.plugins import AudioIdentificationBenchmark


@pytest.fixture(scope="module")
def benchmark():
    return AudioIdentificationBenchmark()


def test_every_labeled_metric_has_an_explanation(benchmark):
    missing = sorted(set(benchmark.metric_labels) - set(benchmark.metric_help))
    assert not missing, "metrics shown with no explanation: {}".format(missing)


def test_no_explanation_describes_a_metric_that_is_not_reported(benchmark):
    """A stale entry is worse than a missing one: it reads as current."""

    orphaned = sorted(set(benchmark.metric_help) - set(benchmark.metric_labels))
    assert not orphaned, "explanations for metrics we do not report: {}".format(orphaned)


def test_the_primary_metric_is_explained(benchmark):
    assert benchmark.primary_metric in benchmark.metric_help


@pytest.mark.parametrize(
    "key",
    sorted(AudioIdentificationBenchmark.metric_help),
)
def test_each_explanation_is_a_real_sentence(key, benchmark):
    text = benchmark.metric_help[key]
    assert len(text) > 60, "{} is too short to explain anything".format(key)
    # Not `isupper`: "1/N for a catalog of N songs" is a correct opening and a
    # digit is not uppercase. What matters is that it is not a sentence
    # fragment continuing from the label.
    assert not text[0].islower(), "{} reads as a fragment, not a sentence".format(key)
    assert text.rstrip().endswith("."), "{} is not a complete sentence".format(key)


@pytest.mark.parametrize(
    "key",
    sorted(AudioIdentificationBenchmark.metric_help),
)
def test_explanations_avoid_benchmark_jargon(key, benchmark):
    """Words that mean something to us and nothing to a student.

    "cell" and "tier" and "grid" are our vocabulary for the corpus layout;
    a student reading a run page has never seen them. "harness" and "driver"
    are the same problem from the other direction.
    """

    jargon = ("grid cell", "the grid", "tier", "harness", "driver", "adapter", "manifest")
    text = benchmark.metric_help[key].lower()
    found = [word for word in jargon if word in text]
    assert not found, "{} explains itself with our vocabulary, not theirs: {}".format(key, found)


def test_the_scored_axes_name_the_course_questions(benchmark):
    """Clip length and noise are the capstone's own analysis axes.

    If these explanations stop referring to what the assignment asks, the
    benchmark has drifted from the course and the drift should be deliberate.
    """

    assert re.search(r"short|length", benchmark.metric_help["short_clip_top1"], re.I)
    assert re.search(r"nois", benchmark.metric_help["noisy_top1"], re.I)
    assert "capstone" in benchmark.metric_help["short_clip_top1"].lower()
    assert "capstone" in benchmark.metric_help["noisy_top1"].lower()


def test_pitch_is_labeled_as_outside_the_assignment(benchmark):
    """The one axis the course never asks for has to say so.

    Pitch shift is where a fingerprint pipeline measurably fails, which makes
    it worth reporting and makes it dishonest to report silently: a team that
    scores badly here did not fail at anything they were asked to do.
    """

    text = benchmark.metric_help["pitch_top1"]
    assert "NOT part of the assignment" in text or "not part of the assignment" in text.lower()


def test_the_two_failure_rates_point_at_different_causes(benchmark):
    """The split is the reason the taxonomy exists; the words must carry it."""

    retrieval = benchmark.metric_help["retrieval_failure_rate"].lower()
    ranking = benchmark.metric_help["ranking_failure_rate"].lower()
    assert "fingerprint" in retrieval
    assert "vot" in ranking or "tally" in ranking
    assert retrieval != ranking


def test_the_trivial_baseline_explains_why_beating_chance_is_not_enough(benchmark):
    text = benchmark.metric_help["trivial_baseline_top1"].lower()
    assert "chance" in text
    assert "fingerprint" in text or "peak" in text
