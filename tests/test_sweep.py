"""The catalog-size sweep, and the assumption that makes it free."""

from __future__ import annotations

import unittest

from audio_identification_benchmark import sweep
from audio_identification_benchmark.datasets import EnrollCase, QueryCase


def enroll(song_id):
    return EnrollCase(song_id=song_id, samples=None, sample_rate=44100)


def query(gold, clip_seconds=10.0):
    return QueryCase(
        query_id="q-{}".format(gold),
        sample_rate=44100,
        clip_seconds=clip_seconds,
        pitch_semitones=0.0,
        snr_db=None,
        gold_song_id=gold,
        kind="in_set",
        source_song_id=gold,
    )


def answer(candidates):
    return {"ok": True, "candidates": list(candidates), "scores": None, "shape": "pairs"}


def catalog(size):
    """``size`` songs named so sorted order is numeric order."""

    return ["song-{:03d}".format(index) for index in range(size)]


class RestrictTests(unittest.TestCase):
    def test_removing_a_higher_ranked_song_promotes_the_gold_song(self):
        """The whole sweep rests on this. Dropping a song the submission
        ranked above gold is what would have happened had that song never
        been enrolled, so gold moves up."""

        ranking = ["song-005", "song-001", "song-009"]
        self.assertEqual(
            sweep.restrict(ranking, ["song-001", "song-009"]),
            ["song-001", "song-009"],
        )

    def test_order_among_survivors_is_unchanged(self):
        ranking = ["a", "b", "c", "d"]
        self.assertEqual(sweep.restrict(ranking, ["d", "b"]), ["b", "d"])


class SweepPointTests(unittest.TestCase):
    def setUp(self):
        self.ids = catalog(40)
        self.cases = [enroll(song_id) for song_id in self.ids]
        self.cases += [query(song_id) for song_id in self.ids]

    def test_a_perfect_submission_is_flat_across_every_size(self):
        outputs = [{"ok": True} for _ in self.ids]
        outputs += [answer([song_id]) for song_id in self.ids]
        points = sweep.sweep_points(outputs, self.cases)

        self.assertTrue(points)
        for point in points:
            self.assertEqual(point["top1"], 1.0)
        self.assertIsNone(sweep.knee(points))

    def test_a_submission_that_degrades_with_size_shows_a_knee(self):
        """Gold ranked second behind a low-numbered song. At a small catalog
        the distractor is present and gold loses; the distractor is in every
        subset because subsets are the lowest ids, so this degrades in the
        opposite direction and proves the curve is not hardcoded."""

        outputs = [{"ok": True} for _ in self.ids]
        for index, song_id in enumerate(self.ids):
            if index < 10:
                outputs.append(answer([song_id]))
            else:
                # A distractor that is in every subset, so these queries are
                # wrong at every size where they are counted at all.
                outputs.append(answer(["song-000", song_id]))
        points = sweep.sweep_points(outputs, self.cases)

        by_size = {point["catalog_size"]: point["top1"] for point in points}
        self.assertEqual(by_size[5], 1.0, "the first five songs all answer correctly")
        self.assertLess(by_size[40], 0.5, "most of the catalog is answered wrong")
        self.assertIsNotNone(sweep.knee(points))

    def test_a_query_whose_song_is_outside_the_subset_is_not_counted(self):
        """Otherwise the curve would fall because of our subsetting rather
        than because of the submission, which would be a lie about their code."""

        outputs = [{"ok": True} for _ in self.ids]
        outputs += [answer([song_id]) for song_id in self.ids]
        points = sweep.sweep_points(outputs, self.cases)

        by_size = {point["catalog_size"]: point["queries"] for point in points}
        self.assertEqual(by_size[5], 5)
        self.assertEqual(by_size[10], 10)
        self.assertEqual(by_size[40], 40)

    def test_points_at_or_above_the_real_catalog_are_dropped(self):
        small = catalog(8)
        cases = [enroll(s) for s in small] + [query(s) for s in small]
        outputs = [{"ok": True} for _ in small] + [answer([s]) for s in small]
        points = sweep.sweep_points(outputs, cases)

        sizes = [point["catalog_size"] for point in points]
        self.assertEqual(sizes, [5, 8], "5 from the ladder, 8 as the full catalog")

    def test_tiny_catalogs_produce_no_curve(self):
        """With four songs, chance alone is 25%. A point there would tell a
        team their pipeline is strong when it is guessing."""

        tiny = catalog(4)
        cases = [enroll(s) for s in tiny] + [query(s) for s in tiny]
        outputs = [{"ok": True} for _ in tiny] + [answer([s]) for s in tiny]
        points = sweep.sweep_points(outputs, cases)

        self.assertEqual([p["catalog_size"] for p in points], [4])
        self.assertIsNone(sweep.describe(points), "one point is not a curve")

    def test_a_failed_query_counts_as_a_retrieval_failure(self):
        outputs = [{"ok": True} for _ in self.ids]
        outputs += [{"ok": False, "error": "boom"} for _ in self.ids]
        points = sweep.sweep_points(outputs, self.cases)

        for point in points:
            self.assertEqual(point["top1"], 0.0)
            self.assertEqual(point["retrieval_failure_rate"], 1.0)


class DescribeTests(unittest.TestCase):
    def test_a_flat_curve_says_nothing_is_size_limited(self):
        points = [
            {"catalog_size": 5, "top1": 0.8, "queries": 5, "retrieval_failure_rate": 0.1},
            {"catalog_size": 40, "top1": 0.79, "queries": 40, "retrieval_failure_rate": 0.1},
        ]
        sentence = sweep.describe(points)
        self.assertIn("holds steady", sentence)

    def test_a_knee_with_high_retrieval_failure_blames_the_fingerprints(self):
        points = [
            {"catalog_size": 5, "top1": 0.9, "queries": 5, "retrieval_failure_rate": 0.0},
            {"catalog_size": 20, "top1": 0.3, "queries": 20, "retrieval_failure_rate": 0.7},
        ]
        sentence = sweep.describe(points)
        self.assertIn("20-song library", sentence)
        self.assertIn("fingerprints rather than the vote", sentence)

    def test_a_knee_with_low_retrieval_failure_blames_the_vote(self):
        points = [
            {"catalog_size": 5, "top1": 0.9, "queries": 5, "retrieval_failure_rate": 0.0},
            {"catalog_size": 20, "top1": 0.3, "queries": 20, "retrieval_failure_rate": 0.05},
        ]
        sentence = sweep.describe(points)
        self.assertIn("the vote is what gives way", sentence)

    def test_a_curve_that_is_flat_at_zero_gets_no_sentence(self):
        """Every template here would be false about a submission that never
        worked, and the diagnostics already say what happened."""

        points = [
            {"catalog_size": 5, "top1": 0.0, "queries": 5, "retrieval_failure_rate": 1.0},
            {"catalog_size": 40, "top1": 0.0, "queries": 40, "retrieval_failure_rate": 1.0},
        ]
        self.assertIsNone(sweep.describe(points))
        self.assertIsNone(sweep.knee(points))


class EquivalenceTests(unittest.TestCase):
    """The sweep's cheapness rests on an assumption about the submission's
    matcher. An assumption that is never checked is a claim."""

    def test_a_per_song_matcher_agrees_with_a_real_re_enrollment(self):
        ids = catalog(10)
        subset = ids[:5]
        cases = [enroll(s) for s in ids] + [query(s) for s in ids]
        full = [{"ok": True} for _ in ids] + [answer(["song-009", s]) for s in ids]
        # What the same matcher returns with only the subset enrolled: the
        # song-009 distractor is gone because it was never in the database.
        actual = [{"ok": True} for _ in ids] + [answer([s]) for s in ids]

        report = sweep.verify_subset_equivalence(full, actual, cases, subset)
        self.assertEqual(report["agreement"], 1.0)
        self.assertEqual(report["disagreements"], 0)
        self.assertEqual(report["queries"], 5)

    def test_a_catalog_wide_matcher_is_reported_as_disagreeing(self):
        ids = catalog(10)
        subset = ids[:5]
        cases = [enroll(s) for s in ids] + [query(s) for s in ids]
        full = [{"ok": True} for _ in ids] + [answer([s]) for s in ids]
        # A matcher whose answer changes with the catalog: with fewer songs
        # enrolled it now names a different one.
        actual = [{"ok": True} for _ in ids] + [answer(["song-001"]) for _ in ids]

        report = sweep.verify_subset_equivalence(full, actual, cases, subset)
        self.assertLess(report["agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
