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
import os
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


def fixture_songs(sample_rate: int = SAMPLE_RATE):
    """Two rendered songs, identical on every machine.

    Rendered rather than recorded so the fixture ships as code, and rendered by
    the same generator the scored corpus uses so a chain that works here works
    there.
    """

    return synth.render_corpus(FIXTURE_SPECS, sample_rate, verify=False)


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
    with _fresh_state():
        return _attempt(fingerprint_chain, enroll_call, query_call, songs, sample_rate)


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
    with tempfile.TemporaryDirectory(prefix="cogworks-accept-") as temporary:
        os.chdir(temporary)
        try:
            yield
        finally:
            os.chdir(previous)


def _attempt(fingerprint_chain, enroll_call, query_call, songs, sample_rate):
    try:
        for song_id, signal in songs.items():
            enroll_call(song_id, _run(fingerprint_chain, signal, sample_rate))
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

    target = "fixture_a"
    clip = fixture_clip(songs[target], sample_rate)
    try:
        answer = query_call(_run(fingerprint_chain, clip, sample_rate))
    except BaseException as error:  # noqa: BLE001
        return False, "querying raised {}: {}".format(
            type(error).__name__, str(error)[:120]
        )

    best = _winner(answer)
    if best == target:
        return True, "identified {} from a two-second clip".format(target)
    return False, "asked for {} and got {}".format(target, best if best else repr(answer)[:60])


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
    """The song a submission named, whatever shape it said it in."""

    if isinstance(answer, str):
        return answer
    try:
        first = next(iter(answer))
    except (TypeError, StopIteration):
        return ""
    if isinstance(first, str):
        return first
    if isinstance(first, (tuple, list)) and first:
        return str(first[0])
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
