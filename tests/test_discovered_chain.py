"""The shapes Week 1 repositories are written in, one per test.

Every case here is a shape the 2026 corpus actually contains, reduced to the
smallest repository that has it. The corpus tests next door prove the numbers;
these prove the search can reach the shape at all, which is the part that
breaks silently when a validator or an arrangement changes.

The repositories are written out rather than imported so each test says what
it is about in one screen. They do real arithmetic on the benchmark's own
fixture, because a stub that returns a constant would pass a search that never
ran anything.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python" / "cogbench" / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "week1"))

import numpy as np  # noqa: E402

from audio_identification_benchmark import roles  # noqa: E402
from audio_identification_benchmark.contracts import SAMPLE_RATE  # noqa: E402
from audio_identification_benchmark.plugins import (  # noqa: E402
    AudioIdentificationBenchmark,
)


#: 252 samples a frame, because 44100 divides by it exactly. The acceptance
#: fixture's clip starts at exactly one second, so an aligned frame is the
#: difference between a query whose offsets line up with the enrolled song and
#: one whose spectra are computed half a frame away from them.
FRAME = 252

#: The fingerprint half every repository below shares. Written once because
#: none of these tests are about it.
PIPELINE = '''
import numpy as np

FRAME = 252


def make_spectrogram(samples, rate):
    frames = len(samples) // FRAME
    block = np.asarray(samples[: frames * FRAME]).reshape(frames, FRAME)
    return np.abs(np.fft.rfft(block, axis=1)).T


def find_peaks(grid):
    cutoff = np.percentile(grid, 99)
    return [(int(f), int(t)) for f, t in np.argwhere(grid > cutoff)]


def make_fingerprints(peaks):
    ordered = sorted(peaks, key=lambda spot: (spot[1], spot[0]))
    found = []
    for index, (f_1, t_1) in enumerate(ordered):
        for f_2, t_2 in ordered[index + 1 : index + 6]:
            found.append(((f_1, f_2, t_2 - t_1), t_1))
    return found
'''

#: Their database is a plain dict their own factory returns, and both halves
#: of the database take it as their first argument. Their query answers with a
#: vote tally, and two more of their own functions turn that into song names.
#: This is rutvim2009 Week1 reduced to its shape.
FACTORY_AND_READERS = PIPELINE + '''

def create_database():
    return {}


def add_fingerprints(database, song_id, fingerprints):
    for key, when in fingerprints:
        database.setdefault(key, []).append((song_id, when))


def query_database(database, fingerprints):
    votes = {}
    for key, when in fingerprints:
        for song_id, at in database.get(key, []):
            votes[(song_id, at - when)] = votes.get((song_id, at - when), 0) + 1
    return votes


def get_sorted_matches(votes):
    return sorted(votes.items(), key=lambda pair: -pair[1])


def get_sorted_songs(matches):
    named = []
    for (song_id, _offset), _votes in matches:
        if song_id not in named:
            named.append(song_id)
    return named
'''

#: Their store takes one fingerprint per call, which is how Cog-gurts wrote
#: `AudioDatabase.add_hash(hash_value, song_id, offset)`. The loop over the
#: song's fingerprints is the one their own database builder writes inline.
ONE_FINGERPRINT_AT_A_TIME = PIPELINE + '''

class Library:
    def __init__(self):
        self.hashes = {}

    def add_hash(self, key, song_id, when):
        self.hashes.setdefault(key, []).append((song_id, when))

    def name_it(self, fingerprints):
        votes = {}
        for key, when in fingerprints:
            for song_id, at in self.hashes.get(key, []):
                votes[(song_id, at - when)] = votes.get((song_id, at - when), 0) + 1
        best = {}
        for (song_id, _offset), count in votes.items():
            best[song_id] = max(best.get(song_id, 0), count)
        return [song_id for song_id, _c in sorted(best.items(), key=lambda p: -p[1])]
'''


#: A song id is a primary key, so enrolling the same song twice is an error
#: their store is right to raise. Their query ranks vote buckets, so one of
#: their own readers is what turns the answer into song names -- and that
#: reader has to be tried against a database this store has not already seen.
REFUSES_A_SONG_TWICE = PIPELINE + '''

class Library:
    def __init__(self):
        self.songs = set()
        self.keys = {}


def create_database():
    return Library()


def add_fingerprints(database, song_id, fingerprints):
    if song_id in database.songs:
        raise ValueError("{} is already enrolled".format(song_id))
    database.songs.add(song_id)
    for key, when in fingerprints:
        database.keys.setdefault(key, []).append((song_id, when))


def query_database(database, fingerprints):
    votes = {}
    for key, when in fingerprints:
        for song_id, at in database.keys.get(key, []):
            votes[(song_id, at - when)] = votes.get((song_id, at - when), 0) + 1
    return sorted(votes.items(), key=lambda pair: -pair[1])


def get_sorted_songs(matches):
    named = []
    for (song_id, _offset), _votes in matches:
        if song_id not in named:
            named.append(song_id)
    return named
'''

#: Their query returns a result object of their own, and its length is the
#: number of songs it has been asked to confirm rather than the number of
#: votes it holds. Reading a length off a class the team wrote says whatever
#: they wrote it to say, which is why the search asks whether the query ran
#: instead.
ANSWERS_WITH_AN_UNREAD_RESULT = PIPELINE + '''

class Result:
    def __init__(self, votes):
        self.votes = votes
        self.confirmed = []

    def __len__(self):
        return len(self.confirmed)


def create_database():
    return {}


def add_fingerprints(database, song_id, fingerprints):
    for key, when in fingerprints:
        database.setdefault(key, []).append((song_id, when))


def query_database(database, fingerprints):
    votes = {}
    for key, when in fingerprints:
        for song_id, at in database.get(key, []):
            votes[(song_id, at - when)] = votes.get((song_id, at - when), 0) + 1
    return Result(votes)


def best_songs(result):
    best = {}
    for (song_id, _offset), count in result.votes.items():
        best[song_id] = max(best.get(song_id, 0), count)
    result.confirmed = [song for song, _c in sorted(best.items(), key=lambda p: -p[1])]
    return result.confirmed
'''


#: Their store is a method that fills a table on its own object, and their
#: matcher is a plain function that has to be handed that table plus a way to
#: turn a song id into a name. Cog-gurts Week 1 reduced to its shape:
#: `AudioDatabase.add_hash(key, id, offset)` fills `self.hash_map`, and
#: `Matcher.match(fp, database, song_index)` returns `song_index[best]`.
STATE_ON_THEIR_OBJECT = PIPELINE + '''

class AudioDatabase:
    def __init__(self):
        self.hash_map = {}
        self.path = "data/db.pkl"

    def add_hash(self, key, song_id, when):
        self.hash_map.setdefault(key, []).append((song_id, when))


class Matcher:
    def match(recording_fp, database, song_index):
        votes = {}
        for key, when in recording_fp:
            for song_id, at in database.get(key, []):
                votes[(song_id, at - when)] = votes.get((song_id, at - when), 0) + 1
        if not votes:
            return None
        (best, _offset), score = max(votes.items(), key=lambda pair: pair[1])
        return {"song": song_index[best], "votes": score}
'''

#: The same repository with a second table their store also fills. Which one
#: their matcher wants is then not something the search can read off the
#: object, so the shape is refused rather than guessed.
TWO_TABLES_ON_THEIR_OBJECT = STATE_ON_THEIR_OBJECT.replace(
    "        self.hash_map.setdefault(key, []).append((song_id, when))",
    "        self.hash_map.setdefault(key, []).append((song_id, when))\n"
    "        self.seen[song_id] = True",
).replace(
    '        self.path = "data/db.pkl"',
    '        self.path = "data/db.pkl"\n        self.seen = {}',
)

#: Their own welded builder, which takes a folder and two tunings. The
#: per-fingerprint arrangement must not offer it a fingerprint as the folder.
A_BUILDER_THAT_TAKES_A_FOLDER = '''
from pathlib import Path


def build_song_database(songs_folder, fanout=15, percentile=75):
    return {str(path): [] for path in Path(str(songs_folder)).glob("*.mp3")}
'''


def _spec():
    return AudioIdentificationBenchmark().discovery()


def _resolved(source: str, tmp: Path, spec=None):
    from cogbench.resolve import from_spec

    (tmp / "theirs.py").write_text(source)
    roles.forget_fingerprints()
    return from_spec(tmp, spec or _spec(), benchmark="audio-identification")


class _InAScratchRepository(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cogworks-week1-shape-")).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.addCleanup(roles.forget_fingerprints)


class TheirDatabaseIsAnArgumentTheirOwnFactoryMakes(_InAScratchRepository):
    """rutvim2009 Week1's shape: `create_database()` returns the dict that
    `add_fingerprints(db, id, fps)` and `query_database(db, fps)` both take
    first, and the answer is two more of their functions past the query."""

    def test_the_factory_the_store_the_query_and_both_readers_all_bind(self):
        found = _resolved(FACTORY_AND_READERS, self.tmp)

        self.assertTrue(found.ready, found.verdict.headline)
        self.assertEqual(found.attempt.enroll, "theirs.add_fingerprints")
        self.assertEqual(found.attempt.query, "theirs.query_database")
        record = found.to_dict()
        self.assertEqual(record["factory"], "theirs.create_database")
        self.assertEqual(
            record["readers"], ["theirs.get_sorted_matches", "theirs.get_sorted_songs"]
        )

    def test_the_week_recognises_their_empty_database_and_nothing_else(self):
        """The predicate the week hands the resolver. Empty is the whole test,
        because a loader that returns yesterday's database would score their
        data rather than the benchmark's."""

        found = _resolved(FACTORY_AND_READERS, self.tmp)
        by_label = {
            step.label: step for step in _every_candidate(found.discovery)
        }

        self.assertTrue(
            roles.looks_like_an_empty_database(by_label["theirs.create_database"])
        )
        self.assertFalse(
            roles.looks_like_an_empty_database(by_label["theirs.get_sorted_matches"])
        )

    def test_a_scored_run_starts_from_a_database_of_their_own_with_nothing_in_it(self):
        """Proving the binding enrolled the two fixture songs into the dict
        their factory made. Scoring from there would rank `fixture_a` against
        the benchmark's real catalog."""

        ready = _resolved(FACTORY_AND_READERS, self.tmp).fresh()
        ready.enroll("gamma", [((1, 2, 3), 4)])

        self.assertEqual(ready.query([((1, 2, 3), 4)]), ["gamma"])


class TheirStoreTakesOneFingerprintAtATime(_InAScratchRepository):
    """Cog-gurts' shape: `add_hash(key, song_id, offset)` is one fingerprint
    per call, so a store offered the whole song raises on its first line."""

    def test_the_per_fingerprint_arrangement_is_the_one_that_binds(self):
        found = _resolved(ONE_FINGERPRINT_AT_A_TIME, self.tmp)

        self.assertTrue(found.ready, found.verdict.headline)
        self.assertEqual(found.attempt.enroll, "theirs.Library().add_hash")
        self.assertEqual(found.attempt.query, "theirs.Library().name_it")
        self.assertGreaterEqual(found.attempt.arrangement, 4)

    def test_it_offers_the_key_and_the_time_of_every_fingerprint(self):
        seen = []
        roles.enroll_arrangements(
            lambda key, song_id, when: seen.append((key, song_id, when)),
            "song",
            [(("a", "b", 1), 7), (("c",), 9)],
        )[4]()

        self.assertEqual(seen, [(("a", "b", 1), "song", 7), (("c",), "song", 9)])

    def test_a_mapping_of_key_to_time_is_read_as_the_fingerprints_it_is(self):
        """Cog-gurts' `fingerprint_recording` returns `{(f1, f2, dt): t}`."""

        seen = []
        roles.enroll_arrangements(
            lambda key, song_id, when: seen.append((key, song_id, when)),
            "song",
            {("a", "b", 1): 7},
        )[4]()

        self.assertEqual(seen, [(("a", "b", 1), "song", 7)])

    def test_the_song_id_may_come_first_instead(self):
        seen = []
        roles.enroll_arrangements(
            lambda key, song_id, when: seen.append((key, song_id, when)),
            "song",
            [(("a",), 7)],
        )[5]()

        self.assertEqual(seen, [("song", ("a",), 7)])

    def test_how_many_arrangements_there_are_is_part_of_the_search_budget(self):
        """Not a count for its own sake. The resolver classifies every one of
        their functions in every arrangement before it pairs anything, so each
        arrangement here is one more whole enrolment of the fixture per
        candidate -- 75,372 calls of a store that takes one fingerprint at a
        time. Anything added here has to be worth that."""

        self.assertEqual(len(roles.enroll_arrangements(lambda *_a: None, "", None)), 6)

    def test_a_row_that_is_not_a_pair_is_refused_rather_than_cut_down(self):
        """Reading the first two fields of a longer row invented a key and a
        time the benchmark had no reason to believe, and their store took them
        without complaint."""

        with self.assertRaises(TypeError):
            roles.enroll_arrangements(
                lambda key, song_id, when: None, "song", [((1, 2, 3), 7, "extra")]
            )[4]()

    def test_fingerprints_that_are_not_pairs_are_a_signature_mismatch(self):
        """Which `accepts` already reads as "try the next arrangement" rather
        than as a broken repository."""

        with self.assertRaises(TypeError):
            roles.enroll_arrangements(
                lambda key, song_id, when: None, "song", [1, 2, 3]
            )[4]()


class ReadingWhatTheirFunctionsReturn(unittest.TestCase):
    """The validators. Loose on purpose, and wrong in exactly two ways that
    the corpus measured."""

    def test_a_pair_whose_key_is_a_tuple_is_a_fingerprint_and_not_a_peak(self):
        """`generate_fingerprints` returns `((1, 262, 9), 0)`. Read as peaks,
        it bound as the peak finder and the search then fingerprinted the
        fingerprints."""

        self.assertFalse(roles.looks_like_peaks([((1, 262, 9), 0), ((3, 4, 1), 2)]))
        self.assertTrue(roles.looks_like_peaks([(1, 262), (3, 4)]))
        self.assertTrue(roles.looks_like_peaks(np.array([[1, 2], [3, 4]])))

    def test_a_mapping_of_key_to_time_is_fingerprints(self):
        """Cog-gurts return `{(f1, f2, dt): t}`; another team could key by an
        int or a string and mean the same thing."""

        self.assertTrue(roles.looks_like_fingerprints({(1, 2, 3): 4}))
        self.assertTrue(roles.looks_like_fingerprints({(1, 2): 4}))
        self.assertTrue(roles.looks_like_fingerprints({"abc": 4}))
        self.assertFalse(roles.looks_like_fingerprints({}))

    def test_a_flat_triple_is_not_a_fingerprint(self):
        """Every Week 1 fingerprinter in the ground truth returns a key and a
        time, four as `((f1, f2, dt), t)` and the fifth as a mapping. Nothing
        there says which field of `(a, b, c)` is which, and the step that
        hands a store one fingerprint at a time needs exactly those two."""

        self.assertFalse(roles.looks_like_fingerprints([(1, 2, 3), (4, 5, 6)]))
        self.assertTrue(roles.looks_like_fingerprints([((1, 2, 3), 0)]))

    def test_a_bare_wide_array_is_the_spectrogram_itself(self):
        self.assertTrue(roles.looks_like_a_whole_spectrogram(np.zeros((129, 700))))
        self.assertFalse(roles.looks_like_a_whole_spectrogram(np.zeros((129, 2))))
        self.assertFalse(
            roles.looks_like_a_whole_spectrogram((np.zeros((129, 700)), [1, 2]))
        )


def _Step(label, call, handoff=None):
    """One bound step, as the search would have recorded it.

    A real `Candidate` rather than a stand-in, because what `_run` reads off
    a step is now the whole call: `Candidate.bound` applies the tuning, the
    per-item loop, and the reading of the previous step's value. A fake with
    only a label and a callable would prove that `_run` calls something, not
    that it makes the call the search proved.
    """

    from cogbench.pipeline import Candidate

    return Candidate(label=label, call=call, module="theirs", handoff=handoff)


class WhichPartOfATupleTheNextStepIsHanded(unittest.TestCase):
    """A step that did two of the week's stages returns both answers. Handing
    the next step the wrong one is not an error, it is a lower score."""

    def test_a_fused_first_step_hands_on_its_peaks_and_not_its_spectrogram(self):
        """rutvim2009's `spectrogram_conversion` returns
        `(log_spectrogram, peaks)` and their `generate_fingerprints` accepts
        either: on the peaks it returns 609 fingerprints and scores 0.547, and
        on the spectrogram it returns 2970 and scores 0.094.

        The search records which of the two it bound, as `handoff`, and `_run`
        replays it. Nothing here re-derives it from the shapes: two readings
        of the same tuple both run, so there is nothing in the value itself to
        choose between them."""

        received = []

        def fused(samples, rate):
            return (np.zeros((1025, 171)), [(0, 1), (0, 51), (3, 4)])

        def fingerprints(value):
            received.append(value)
            return [((1, 2, 3), 0)]

        roles._run(
            (
                _Step("theirs.fused", fused),
                _Step("theirs.fingerprints", fingerprints, handoff="element:1"),
            ),
            np.zeros(1000, dtype=np.float32),
            SAMPLE_RATE,
        )

        self.assertEqual(received, [[(0, 1), (0, 51), (3, 4)]])

    def test_a_step_per_stage_is_handed_the_whole_value_as_before(self):
        received = []

        def spectrogram(samples, rate):
            return np.zeros((129, 700))

        def peaks(grid):
            received.append("peaks")
            return [(1, 2), (3, 4)]

        def fingerprints(found):
            received.append(found)
            return [((1, 2, 3), 0)]

        roles._run(
            (
                _Step("theirs.spectrogram", spectrogram),
                _Step("theirs.peaks", peaks),
                _Step("theirs.fingerprints", fingerprints),
            ),
            np.zeros(1000, dtype=np.float32),
            SAMPLE_RATE,
        )

        self.assertEqual(received, ["peaks", [(1, 2), (3, 4)]])


class AQueryThatHasNotRunYet(unittest.TestCase):
    """`fanout_pairs` is a generator function, so a pairing that tries it as
    the query returns something that runs their code when it is read."""

    def test_reading_a_lazy_answer_is_inside_the_guard(self):
        def lazy(_item):
            def rows():
                raise ValueError("too many values to unpack")
                yield  # pragma: no cover - unreachable, and needed to be lazy

            return rows()

        songs = roles.fixture_songs(SAMPLE_RATE)
        chain = (_Step("theirs.fingerprints", lambda samples, rate: [((1, 2), 0)]),)
        self.addCleanup(roles.forget_fingerprints)

        graded, detail = roles.accepts(chain, lambda *_a: None, lazy)

        self.assertFalse(graded)
        self.assertIn("querying raised ValueError", detail)
        self.assertNotIn("fixture", songs and detail)


class EveryBindingRunsItsOwnCall(unittest.TestCase):
    """The fixture fingerprints are computed once and kept, because one
    repository tries 3962 store-and-query pairings against one already-bound
    chain. What "one chain" means is the whole call, not the list of names:
    two bindings can agree on every label and disagree on which function
    object, which tuning, or which reading of the upstream value they run
    with, and the second was handed the first one's answer without ever
    running."""

    def setUp(self):
        self.addCleanup(roles.forget_fingerprints)

    def test_the_same_name_over_a_different_function_is_a_different_answer(self):
        """Two objects of one class give their methods one label between
        them."""

        songs = roles.fixture_songs(SAMPLE_RATE)
        first = (_Step("theirs.Library().fingerprints", lambda samples, rate: [((1, 2, 3), 0)]),)
        second = (_Step("theirs.Library().fingerprints", lambda samples, rate: [((9, 8, 7), 5)]),)

        _enrolled, asked_first = roles._fingerprinted(first, songs, SAMPLE_RATE)
        _enrolled, asked_second = roles._fingerprinted(second, songs, SAMPLE_RATE)

        self.assertEqual(asked_first, [((1, 2, 3), 0)])
        self.assertEqual(asked_second, [((9, 8, 7), 5)])

    def test_the_same_functions_read_two_ways_are_two_answers(self):
        """Same labels, same function objects, one `handoff` apart: the
        difference between fingerprinting a team's spectrogram and
        fingerprinting their peaks, which is 0.094 against 0.547 on the
        corpus."""

        songs = roles.fixture_songs(SAMPLE_RATE)

        def fused(samples, rate):
            return ("the whole spectrogram", [((1, 2, 3), 0)])

        def onward(value):
            return value

        def chain(handoff):
            return (
                _Step("theirs.fused", fused),
                _Step("theirs.onward", onward, handoff=handoff),
            )

        _enrolled, first = roles._fingerprinted(chain("element:0"), songs, SAMPLE_RATE)
        _enrolled, second = roles._fingerprinted(chain("element:1"), songs, SAMPLE_RATE)

        self.assertEqual(first, "the whole spectrogram")
        self.assertEqual(second, [((1, 2, 3), 0)])


class HowCompletelyAPairingAnsweredIsReadTheWayTheScorerReadsIt(unittest.TestCase):
    """`_winner` looks deeper into a value than the benchmark's scorer does.
    A pairing accepted on what `_winner` found alone is one the scorer reads
    as nothing at all."""

    def setUp(self):
        self.addCleanup(roles.forget_fingerprints)

    def _accepts(self, query):
        chain = (_Step("theirs.fingerprints", lambda samples, rate: [((1, 2), 0)]),)
        return roles.accepts(chain, lambda *_a: None, query)

    def test_an_answer_the_scorer_throws_away_is_not_a_pass(self):
        """Asterisk's `rank_songs`: two rows, a song id in each, and a score
        beside it that `_as_float` cannot read. All 68 of that binding's
        queries were rejected, for an identification score of 0.0."""

        graded, detail = self._accepts(
            lambda _item: [("fixture_a", {"votes": 3}), ("fixture_b", {"votes": 1})]
        )

        self.assertFalse(graded)
        self.assertIn("cannot read", detail)

    def test_vote_buckets_are_not_a_pass_either(self):
        """rutvim2009's `get_sorted_matches`, one function short of their own
        `get_sorted_songs`. Refusing it is what leaves the reader search free
        to go on and find that function."""

        graded, _detail = self._accepts(
            lambda _item: [(("fixture_a", 43), 67), (("fixture_b", 12), 2)]
        )

        self.assertFalse(graded)

    def test_a_bare_winning_id_is_still_worth_half(self):
        """The shape the half grade is for: a team whose query names the song
        and stops. The benchmark reads it, and reads one song out of it."""

        graded, _detail = self._accepts(lambda _item: "fixture_a")

        self.assertEqual(graded, 0.5)

    def test_a_ranking_the_scorer_reads_is_still_worth_all_of_it(self):
        graded, _detail = self._accepts(lambda _item: ["fixture_a", "fixture_b"])

        self.assertEqual(graded, 1.0)


class NamesOrderTheSearchAndDoNotDecideIt(_InAScratchRepository):
    """Every function in `FACTORY_AND_READERS` is named the way the course
    names it, so a search that read names would bind it and look right.

    `Stage.prefers` sorts the candidates a stage tries, which is all it is
    for. Emptying it is the only way to tell a chain that was chosen by
    running their code from one that was chosen by spelling, because from the
    outside the two are the same chain.
    """

    def test_the_repository_scores_the_same_with_the_preferences_emptied(self):
        named = _resolved(FACTORY_AND_READERS, self.tmp)
        blind = _resolved(FACTORY_AND_READERS, self.tmp, _blind_spec())

        self.assertTrue(blind.ready, blind.verdict.headline)
        # The score rather than the chain: which of their functions the
        # frontier reaches first is allowed to move, and what the repository
        # is worth is not.
        with_names, without = _score(named), _score(blind)

        self.assertEqual(round(without, 6), round(with_names, 6))
        # Both binding the wrong chain would satisfy the line above. The
        # chain that names decided on scores 0.125.
        self.assertGreater(with_names, 0.5)


class ATailOfReadersIsTriedAgainstAnEmptyDatabase(_InAScratchRepository):
    """Their store refuses a song it already holds, which is ordinary: an id
    is a primary key. Each tail of readers therefore has to be tried against a
    database of its own."""

    def test_the_reader_binds_even_though_the_store_refuses_a_repeat(self):
        found = _resolved(REFUSES_A_SONG_TWICE, self.tmp)

        self.assertTrue(found.ready, found.verdict.headline)
        self.assertEqual(found.to_dict()["readers"], ["theirs.get_sorted_songs"])

    def test_their_store_really_does_refuse_the_second_enrolment(self):
        """Without this the test above would pass for the wrong reason."""

        ready = _resolved(REFUSES_A_SONG_TWICE, self.tmp).fresh()
        ready.enroll("gamma", [((1, 2, 3), 4)])

        with self.assertRaises(ValueError):
            ready.enroll("gamma", [((1, 2, 3), 4)])


class AQueryIsWorthReadingFurtherWheneverItRan(_InAScratchRepository):
    """Their answer is an object of their own that reports a length of zero
    until something reads it. `len(answer) > 0` is a question about a method
    they wrote, and it answered no for a value that holds the whole tally."""

    def test_a_reader_is_tried_on_an_answer_that_reports_no_length(self):
        found = _resolved(ANSWERS_WITH_AN_UNREAD_RESULT, self.tmp)

        self.assertTrue(found.ready, found.verdict.headline)
        self.assertEqual(found.to_dict()["readers"], ["theirs.best_songs"])


class TheirMatcherIsHandedTheTableTheirStoreFilled(_InAScratchRepository):
    """Cog-gurts' shape: the database is neither an argument their store took
    nor a module global, but state on the object their store is a method of,
    and their matcher is a pure function that takes it."""

    def test_the_store_the_matcher_and_both_supplied_values_bind(self):
        found = _resolved(STATE_ON_THEIR_OBJECT, self.tmp)

        self.assertTrue(found.ready, found.verdict.headline)
        record = found.to_dict()
        self.assertEqual(record["enroll"], "theirs.AudioDatabase().add_hash")
        self.assertEqual(record["query"], "theirs.Matcher.match")

    def test_the_record_says_what_the_benchmark_handed_their_matcher(self):
        record = _resolved(STATE_ON_THEIR_OBJECT, self.tmp).to_dict()

        supplied = [row["supplied"] for row in record.get("supplied", ())]
        self.assertIn("hash_map", supplied)
        self.assertIn("an id-to-name table over the enrolled songs", supplied)

    def test_a_scored_run_reads_the_table_off_the_object_it_built(self):
        """Not off the one the search filled. The search enrolled `fixture_a`
        and `fixture_b` to prove the binding; a scored run that answered out
        of that object would rank the fixture songs against the real catalog."""

        ready = _resolved(STATE_ON_THEIR_OBJECT, self.tmp).fresh()
        ready.enroll("gamma", [((1, 2, 3), 4)])

        answer = ready.query([((1, 2, 3), 4)])

        self.assertEqual(answer["song"], "gamma")
        self.assertEqual(answer["votes"], 1)

    def test_two_filled_tables_are_refused_rather_than_guessed(self):
        """Which of their tables their matcher wants is not readable off the
        object once their store fills two of them."""

        found = _resolved(TWO_TABLES_ON_THEIR_OBJECT, self.tmp)

        self.assertFalse(found.ready)
        self.assertIn("database", found.verdict.headline.lower())

    def test_the_refusal_names_both_tables(self):
        from cogbench.resolve import AmbiguousStore, _FromTheirStore

        class Theirs:
            def __init__(self):
                self.hash_map = {"a": 1}
                self.seen = {"b": 2}

            def add(self, key, song_id, when):
                pass

        state = _FromTheirStore(Theirs().add)
        state.enrolling("gamma")

        with self.assertRaises(AmbiguousStore) as raised:
            state.arguments()

        self.assertIn("hash_map", str(raised.exception))
        self.assertIn("seen", str(raised.exception))


class OneFingerprintAtATimeIsOfferedOnlyToAStoreThatTakesThree(unittest.TestCase):
    """Their own welded builder takes a folder and two tunings, and Python
    lets a fingerprint land in the folder. It raises nothing and reads a
    directory, once per fingerprint.

    The question is `resolve._takes_n(store, 3)`, which is the same one the
    resolver asks of a query it is about to hand three arguments. Exactly
    three required positionals: a `*args` store and a store whose offset has a
    default are both refused, and no repository under `.cache/student-repos`
    has either."""

    def test_a_store_requiring_one_argument_is_not_a_per_fingerprint_store(self):
        scope: dict = {}
        exec(A_BUILDER_THAT_TAKES_A_FOLDER, scope)

        with self.assertRaises(TypeError):
            roles.enroll_arrangements(
                scope["build_song_database"], "song", [(("a", "b", 1), 7)]
            )[4]()

    def test_a_store_requiring_three_still_gets_every_fingerprint(self):
        seen = []

        def add_hash(key, song_id, when):
            seen.append((key, song_id, when))

        roles.enroll_arrangements(add_hash, "song", [(("a",), 7), (("b",), 9)])[4]()

        self.assertEqual(seen, [(("a",), "song", 7), (("b",), "song", 9)])

    def test_a_store_that_cannot_be_called_with_three_is_still_refused(self):
        """Both halves of the rule. The one above relaxes what counts as
        required; this one is the part that must not move."""

        with self.assertRaises(TypeError):
            roles.enroll_arrangements(
                lambda key, song_id: None, "song", [(("a",), 7)]
            )[4]()


class WhatCountsAsRankingMoreThanOneSong(unittest.TestCase):
    """Read down the path scoring uses, because the search is predicting what
    the scorer will read and any second opinion can disagree with it. The ids
    themselves rather than how many, because both grades `accepts` gives are
    read off this list: one that holds the song is worth half, one that holds
    more than one song is worth all of it, and one that holds nothing is a
    pairing the benchmark would score at zero."""

    def test_a_list_of_song_ids_ranks_more_than_one(self):
        self.assertEqual(
            roles._scored_ids(["fixture_a", "fixture_b"]), ["fixture_a", "fixture_b"]
        )

    def test_their_winner_and_their_tally_ranks_more_than_one(self):
        """carti4ce's `query_details`: the name, the count, and the votes
        behind it, whose buckets `_ranking` folds into the lower ranks."""

        answer = ("fixture_a", 67, {("fixture_a", 43): 67, ("fixture_b", 12): 2})

        self.assertEqual(roles._scored_ids(answer), ["fixture_a", "fixture_b"])

    def test_ranked_vote_buckets_do_not_rank_more_than_one(self):
        """rutvim2009's `get_sorted_matches`: eight rows and no song id in the
        place `coerce_candidates` reads one, so the scorer throws it out."""

        answer = [(("fixture_a", 43), 67), (("fixture_a", 44), 11), (("fixture_b", 12), 2)]

        self.assertEqual(roles._scored_ids(answer), [])

    def test_ids_paired_with_dictionaries_do_not_rank_more_than_one(self):
        """Asterisk's shape. Two rows, a song id in each, and a score
        `_as_float` cannot read, so `coerce_candidates` refuses the list."""

        answer = [("fixture_a", {"votes": 3}), ("fixture_b", {"votes": 1})]

        self.assertEqual(roles._scored_ids(answer), [])

    def test_one_id_and_its_score_does_not_rank_more_than_one(self):
        """A list of length two with a string in front of it, and one song."""

        self.assertEqual(roles._scored_ids(("fixture_a", 42)), ["fixture_a"])

    def test_one_song_id_does_not_rank_more_than_one(self):
        self.assertEqual(roles._scored_ids("fixture_a"), ["fixture_a"])


class TheGradeIsReadOffTheAnswerAndNothingElse(unittest.TestCase):
    """The resolver grades one of their readers by handing `grades` what the
    reader returned, with no store, no query and no database behind it. That
    is the same judgement `accepts` makes only if `accepts` makes it out of
    the answer alone."""

    def setUp(self):
        self.addCleanup(roles.forget_fingerprints)
        self.chain = (
            _Step("theirs.fingerprints", lambda samples, rate: [((1, 2, 3), 0)]),
        )

    def _end_to_end(self, answer):
        """The grade a pairing whose query returned ``answer`` is given."""

        return roles.accepts(self.chain, lambda *_a: None, lambda _item: answer)

    def test_every_grade_is_the_one_the_answer_alone_earns(self):
        for answer in (
            ["fixture_a", "fixture_b"],
            "fixture_a",
            ("fixture_a", 42),
            [("fixture_a", {"votes": 3}), ("fixture_b", {"votes": 1})],
            "fixture_b",
            [],
        ):
            with self.subTest(answer=answer):
                self.assertEqual(roles.grades(answer), self._end_to_end(answer))

    def test_a_lazy_answer_that_raises_when_read_is_a_refusal_not_an_escape(self):
        """`Cog-gurts`'s `fanout_pairs` is a generator function, and reading
        what one returned runs their code. A reader is handed such a value
        outside any acceptance test, so the guard has to be here."""

        def rows():
            raise ValueError("too many values to unpack")
            yield  # pragma: no cover - unreachable, and needed to be lazy

        graded, detail = roles.grades(rows())

        self.assertFalse(graded)
        self.assertIn("querying raised ValueError", detail)


class EnrollingIsAskedOnItsOwn(unittest.TestCase):
    """Whether a store takes the fixture is a property of the store, so the
    resolver asks it once per store rather than once per store and query.
    `query_call=None` is that question."""

    def setUp(self):
        self.addCleanup(roles.forget_fingerprints)
        self.chain = (
            _Step("theirs.fingerprints", lambda samples, rate: [((1, 2, 3), 0)]),
        )

    def test_a_store_that_took_both_songs_passes_without_being_queried(self):
        kept = []

        graded, detail = roles.accepts(
            self.chain, lambda song_id, prints: kept.append(song_id), None
        )

        self.assertTrue(graded)
        self.assertIn("enrolled", detail)
        self.assertEqual(kept, ["fixture_a", "fixture_b"])

    def test_a_store_that_raised_fails_the_same_way_it_always_did(self):
        def refuse(song_id, prints):
            raise ValueError("this one never takes a song")

        graded, detail = roles.accepts(self.chain, refuse, None)

        self.assertFalse(graded)
        self.assertIn("enrolling raised ValueError", detail)

    def test_a_store_the_arrangement_does_not_fit_is_a_signature_mismatch(self):
        """Which `accepts` reads as "try the next arrangement" rather than as
        a broken repository, exactly as it does with a query."""

        graded, detail = roles.accepts(self.chain, lambda only_one: None, None)

        self.assertFalse(graded)
        self.assertIn("did not accept those arguments", detail)


class ALoaderIsNotAFactoryEvenWhenItsSongsArePrivate(unittest.TestCase):
    """The factory question is whether the thing arrives empty. An attribute
    named `_songs` is as full as one named `songs`, and scoring against it
    would score their data instead of the benchmark's."""

    def test_a_database_whose_songs_live_behind_an_underscore_is_refused(self):
        class Loaded:
            def __init__(self) -> None:
                self._songs = {"fixture_a": [(1, 2, 3)]}
                self._hashes = {(1, 2, 3): [("fixture_a", 0)]}

        self.assertFalse(roles.looks_like_an_empty_database(Loaded))

    def test_the_same_database_with_those_two_empty_is_a_factory(self):
        class Empty:
            def __init__(self) -> None:
                self._songs = {}
                self._hashes = {}

        self.assertTrue(roles.looks_like_an_empty_database(Empty))


def _blind_spec():
    """The same week with every name hint removed from every stage."""

    from dataclasses import replace

    spec = _spec()
    role = spec.chain_role
    return replace(
        spec,
        chain_role=replace(
            role, stages=tuple(replace(stage, prefers=()) for stage in role.stages)
        ),
    )


def _score(found) -> float:
    """What the week's own scorer makes of a binding, end to end."""

    from audio_identification_benchmark.discovered import build

    plugin = AudioIdentificationBenchmark()
    cases = plugin.load_cases("test")
    resources = plugin.model_factory()
    previous = os.getcwd()
    os.chdir(resources.scratch_dir)
    try:
        ran = plugin.run(lambda *a, **k: build(found), resources, cases)
        return float(plugin.score(ran, cases)["identification_score"])
    finally:
        os.chdir(previous)


def _every_candidate(discovery):
    from cogbench.pipeline import callables_in, instances_in, methods_of

    found = list(callables_in(discovery.namespace))
    for label, instance in instances_in(discovery.namespace):
        found.extend(methods_of(label, instance))
    return found


if __name__ == "__main__":
    unittest.main()
