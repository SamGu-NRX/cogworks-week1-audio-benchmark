"""What Week 1 asks for, described so a repository can be searched for it.

The capstone is one task: put songs in a database, then name the song a short
clip came from. The course names the stages on the way there -- spectrogram,
peaks, fanout fingerprints, database, query -- and every team writes them, in
their own files, under their own names.

So this file says what each stage does in terms of what goes in and what comes
back, and never in terms of what it is called. ``cogbench.pipeline`` searches a
repository against that description by running candidates and feeding each
one's real output to the next, and the acceptance test at the bottom is the
only thing that can accept a chain: enroll two songs the benchmark rendered,
query a clip cut from one of them, and require that song back.

The validators here are deliberately loose. They exist to cut the search, not
to judge, and the corpus shows why: peaks come back as an ``(N, 2)`` array from
one team and a list of tuples from another, and a spectrogram arrives bare from
one and as ``(spec, freqs, times)`` from the next. A validator tight enough to
name one of those the right answer would cost the other team its score, which
is a worse error than a slower search. Only the acceptance test decides.
"""

from __future__ import annotations

import contextlib
import copy
import io
import os
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

from cogbench.pipeline import Role, Stage

from . import synth
from .contracts import SAMPLE_RATE

__all__ = [
    "FINGERPRINT_ROLE",
    "IDENTIFY_ROLE",
    "accepts",
    "enroll_arrangements",
    "fixture_songs",
    "fixture_clip",
    "looks_like_spectrogram",
    "looks_like_peaks",
    "looks_like_fingerprints",
    "looks_like_ranking",
    "forget_fingerprints",
]

#: Long enough that a peak picker has something to find. A pure tone yields
#: zero peaks under every team's threshold, and a fixture that finds nothing
#: would refuse working code.
FIXTURE_SECONDS = 4.0

#: Two songs is the smallest corpus where naming the right one is not luck.
FIXTURE_SPECS = (
    synth.SongSpec(song_id="fixture_a", seed=20260819, duration_seconds=FIXTURE_SECONDS),
    synth.SongSpec(song_id="fixture_b", seed=31337, duration_seconds=FIXTURE_SECONDS),
)


#: The rendered fixture, kept after the first render.
#: Rendering is deliberately exact: `exactmath` computes its own sin and exp so
#: every machine produces identical samples, and that costs about 60ms a song.
#: A search that tries 3962 pairings rendered them 3962 times for 248 seconds
#: of a 352-second check, all of it recomputing a constant.
_RENDERED: dict = {}


def fixture_songs(sample_rate: int = SAMPLE_RATE):
    """Two rendered songs, identical on every machine.

    Rendered rather than recorded so the fixture ships as code, and rendered by
    the same generator the scored corpus uses so a chain that works here works
    there.

    The arrays are handed out read-only, because they are now shared. A student
    function that writes into its input would otherwise change what every later
    attempt is asked about, and the search would start depending on its own
    order.
    """

    if sample_rate not in _RENDERED:
        songs = synth.render_corpus(FIXTURE_SPECS, sample_rate, verify=False)
        for signal in songs.values():
            signal.flags.writeable = False
        _RENDERED[sample_rate] = songs
    return _RENDERED[sample_rate]


def fixture_clip(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A clip from the middle of a song, the way a real query arrives.

    Taken from an offset rather than the start, because a fingerprint scheme
    that only matches at time zero is a real bug this catches.
    """

    start = int(sample_rate * 1.0)
    stop = start + int(sample_rate * 2.0)
    return np.ascontiguousarray(signal[start:stop])


# --------------------------------------------------------------------------
# What each stage's output looks like, loosely
# --------------------------------------------------------------------------


def _first_array(value: Any) -> Any:
    """The array in a value that may be one, or a tuple beginning with one."""

    if isinstance(value, tuple) and value:
        return value[0]
    return value


def looks_like_spectrogram(value: Any) -> bool:
    """Two-dimensional, with more than one frame.

    The frame count is what separates a spectrogram from an array a peak
    finder returns when handed raw audio: one audited repository's
    ``find_peaks`` accepts samples and returns shape ``(440, 1)``, which is
    two-dimensional and is not a spectrogram.
    """

    array = _first_array(value)
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        return False
    return array.shape[0] > 1 and array.shape[1] > 1


def looks_like_peaks(value: Any) -> bool:
    """A sequence of pairs, however a team stores them.

    One team returns ``(N, 2)`` from ``np.argwhere``; another returns a list of
    ``(int, int)`` tuples. Both are peaks.
    """

    if isinstance(value, np.ndarray):
        return value.ndim == 2 and value.shape[0] > 0 and value.shape[1] == 2
    try:
        first = next(iter(value))
    except (TypeError, StopIteration):
        return False
    try:
        return len(first) == 2
    except TypeError:
        return False


def looks_like_fingerprints(value: Any) -> bool:
    """Pairs of (key, time), or triples, with a hashable key.

    The key is what goes in the database, so it has to be hashable; that single
    property separates a fingerprint list from a peak list without caring
    whether the key is a tuple, an int, or a string.
    """

    try:
        first = next(iter(value))
    except (TypeError, StopIteration):
        return False
    if isinstance(first, (tuple, list)) and len(first) == 2:
        try:
            hash(first[0])
        except TypeError:
            return False
        return True
    return isinstance(first, (tuple, list)) and len(first) == 3


def looks_like_ranking(value: Any) -> bool:
    """Something that names songs: ids, or (id, score) pairs, or one id.

    A team that returns only the winner and a team that returns a ranked list
    are both answering the question. Which one they did is read off the value,
    never required in advance.
    """

    if isinstance(value, str):
        return True
    if isinstance(value, dict):
        return any(looks_like_ranking(entry) for entry in value.values())
    try:
        first = next(iter(value))
    except (TypeError, StopIteration):
        return False
    if isinstance(first, str):
        return True
    return isinstance(first, (tuple, list)) and len(first) >= 1


# --------------------------------------------------------------------------
# The roles
# --------------------------------------------------------------------------

#: Samples in, fingerprints out. This is the half of the capstone that both
#: enrolling and querying share, which is why it is resolved once and used
#: twice: a team whose enroll and identify disagree about fingerprinting has a
#: bug, and running the same chain for both is how that bug stays visible.
FINGERPRINT_ROLE = Role(
    "fingerprint",
    (
        Stage(
            "spectrogram",
            prefers=("spectrogram", "spectogram", "stft", "specgram"),
            produces=looks_like_spectrogram,
            arity=2,
        ),
        Stage(
            "peaks",
            prefers=("peak", "local_max", "maxima"),
            produces=looks_like_peaks,
        ),
        Stage(
            "fingerprints",
            prefers=("fingerprint", "fgp", "fanout", "hash", "pair"),
            produces=looks_like_fingerprints,
        ),
    ),
)

#: Samples in, song ids out. Resolved separately because a team that wrote one
#: function for the whole thing is answering the question too, and their
#: ``identify(samples, rate)`` binds here in a single stage.
IDENTIFY_ROLE = Role(
    "identify",
    (
        Stage(
            "identify",
            prefers=("identify", "recognize", "match", "query", "search"),
            produces=looks_like_ranking,
            arity=2,
        ),
    ),
)


# --------------------------------------------------------------------------
# Acceptance: the only thing that may accept a chain
# --------------------------------------------------------------------------


def accepts(fingerprint_chain, enroll_call, query_call, sample_rate: int = SAMPLE_RATE):
    """Enroll two songs, query a clip of one, and require that song back.

    This is deliberately the weakest test that still kills a wrong binding. It
    is not a measure of quality: a team whose fanout is badly tuned should be
    scored badly by the benchmark, not refused by discovery. Two songs, one
    exact excerpt, right answer at rank 1 -- anything wired correctly passes,
    and a chain assembled from the wrong functions does not.

    Returns ``(passed, detail)``. The detail says what was asked and what came
    back, which is what a report shows when this is the step that failed.
    """

    songs = fixture_songs(sample_rate)
    try:
        prints = _fingerprinted(fingerprint_chain, songs, sample_rate)
    except BaseException as error:  # noqa: BLE001 - student code raises anything
        return False, "fingerprinting raised {}: {}".format(
            type(error).__name__, str(error)[:120]
        )
    with _fresh_state():
        return _attempt(enroll_call, query_call, prints)


#: Fingerprints for the fixture, keyed by the chain that produced them.
#: One repository tries 3962 store-and-query pairings, and the chain is already
#: bound by then: the same two songs and the same clip go through the same
#: three functions every time, for the same answer. Running them once took a
#: 278-second check down to 21. Keyed on the chain because a different chain is
#: a different answer, and cleared per resolution because holding a student's
#: arrays after their run is memory nobody asked us to keep.
_FINGERPRINTED: dict = {}

#: The clip is cut from this song, so this is the id a correct binding names.
TARGET = "fixture_a"


def _fingerprinted(chain: Sequence[Any], songs, sample_rate: int):
    """Fingerprints for both fixture songs and the query clip, computed once.

    Their fingerprinting is pure with respect to the search: it turns audio
    into fingerprints and the search never changes the audio. Whatever it
    writes to disk while doing so happens under `_fresh_state` the first time
    and is not part of what enrolling reads back, because enrolling is what
    gets the fresh directory per attempt.
    """

    key = tuple(step.label for step in chain)
    if key not in _FINGERPRINTED:
        with _fresh_state():
            enrolled = [
                (song_id, _run(chain, signal, sample_rate))
                for song_id, signal in songs.items()
            ]
            asked = _run(chain, fixture_clip(songs[TARGET], sample_rate), sample_rate)
        _FINGERPRINTED[key] = (enrolled, asked)
    return _FINGERPRINTED[key]


def forget_fingerprints() -> None:
    """Drop the cached fixture fingerprints.

    Called when a resolution finishes. Their arrays can be large and there is
    no reason to hold them after the search that needed them is over.
    """

    _FINGERPRINTED.clear()


@contextlib.contextmanager
def _fresh_state():
    """One attempt, one empty world.

    A student database is often a file: one audited repository pickles to a
    relative ``db.pkl`` and reloads it on every call. Without a fresh directory
    per attempt, a failed pairing leaves songs enrolled that the next pairing
    then finds, and the search reports whichever attempt happened to run after
    a good one. Each attempt gets its own directory and its own copy of every
    module-level container the previous attempt may have filled.
    """

    previous = os.getcwd()
    saved_out, saved_err = sys.stdout, sys.stderr
    with tempfile.TemporaryDirectory(prefix="cogworks-accept-") as temporary:
        os.chdir(temporary)
        # Their functions narrate: one prints every fingerprint it builds,
        # thousands of lines per call. That output belongs to their run, not
        # to the search for which of their functions to call.
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            yield
        finally:
            sys.stdout, sys.stderr = saved_out, saved_err
            os.chdir(previous)


def _attempt(enroll_call, query_call, prints):
    # A copy per attempt, because the fingerprints are computed once and handed
    # to thousands of candidate stores. A store that sorts its argument in place
    # or pops from it would otherwise change what later attempts are given, and
    # the search would start depending on its own order.
    #
    # Shallow, not deep. Fingerprints are a list of immutable tuples, and
    # copying five thousand of those element by element cost 100 seconds of a
    # 352-second check. What has to be private is the container a store might
    # mutate, and `_shallow` copies exactly that.
    enrolled = [(song_id, _shallow(value)) for song_id, value in prints[0]]
    asked = _shallow(prints[1])
    try:
        for song_id, fingerprints in enrolled:
            enroll_call(song_id, fingerprints)
    except TypeError as error:
        # A signature mismatch is not a wrong function, it is a different
        # arrangement of the same one. The caller retries; see
        # `enroll_arrangements`.
        return False, "enrolling did not accept those arguments: {}".format(
            str(error)[:120]
        )
    except BaseException as error:  # noqa: BLE001 - student code raises anything
        return False, "enrolling raised {}: {}".format(
            type(error).__name__, str(error)[:120]
        )

    target = TARGET
    try:
        answer = query_call(asked)
    except BaseException as error:  # noqa: BLE001
        return False, "querying raised {}: {}".format(
            type(error).__name__, str(error)[:120]
        )

    best = _winner(answer)
    if best == target:
        return True, "identified {} from a two-second clip".format(target)
    return False, "asked for {} and got {}".format(target, best if best else repr(answer)[:60])


def _shallow(value: Any) -> Any:
    """A private copy of the container, sharing whatever is inside it.

    Teams return fingerprints as a list of tuples, a numpy array, or a dict of
    those. Each of those is copied one level down, which is the level a store
    can mutate. The tuples inside cannot be changed by anyone, so copying them
    protects nothing and costs the search a third of its running time.
    """

    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _run(chain: Sequence[Any], signal: np.ndarray, sample_rate: int) -> Any:
    """Push audio through a resolved chain, changing nothing on the way."""

    value = chain[0].call(signal, sample_rate)
    for step in chain[1:]:
        try:
            value = step.call(value)
        except BaseException:  # noqa: BLE001
            # The same unpacking the search used: their spectrogram may be a
            # tuple and their peak finder may want the array inside it.
            if isinstance(value, tuple) and value:
                value = step.call(value[0])
            else:
                raise
    return value


def _winner(answer: Any) -> str:
    """The song a submission named, whatever shape it said it in.

    Teams return a bare id, a ranked list, a list of ``(id, score)`` pairs, or
    a dict holding several of those at once. One repository's ``query`` returns
    ``{"best_matches": ..., "ranked": [...], "offsets": ...}``, and reading the
    first key of that dict yields ``best_matches``, which is not a song. So a
    dict is unwrapped by looking for the entry that holds a ranking rather than
    by taking whatever comes first.
    """

    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        for value in answer.values():
            named = _winner(value)
            if named:
                return named
        return ""
    try:
        first = next(iter(answer))
    except (TypeError, StopIteration):
        return ""
    if isinstance(first, str):
        return first
    if isinstance(first, (tuple, list)) and first:
        return _winner(first[0]) if not isinstance(first[0], str) else first[0]
    return ""


def enroll_arrangements(store, song_id: str, fingerprints: Any):
    """Every way a store might want one song offered to it.

    Teams do not agree on this and there is no reason they should. One writes
    ``add(fingerprints, song_id)``; another writes
    ``add(fanout, song_ID, song_name)`` because their database keeps a title
    beside the id; a third takes the id first. Rather than requiring one of
    those, the acceptance test tries each and lets the query decide: an
    arrangement that stored the song wrongly cannot answer with it.

    The song id is passed for every argument that wants a name, because the id
    is the only string the benchmark has and the only one it will accept back.
    """

    return (
        lambda: store(fingerprints, song_id),
        lambda: store(song_id, fingerprints),
        lambda: store(fingerprints, song_id, song_id),
        lambda: store(song_id, song_id, fingerprints),
    )
