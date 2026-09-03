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
import inspect
import io
import os
import sys
import tempfile
from types import MemberDescriptorType
from typing import Any, List, Sequence

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
    "looks_like_a_whole_spectrogram",
    "looks_like_fingerprints",
    "looks_like_ranking",
    "looks_like_an_empty_database",
    "forget_fingerprints",
    "PER_FINGERPRINT",
    "POST_QUERY_READERS",
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
    """A sequence of pairs of numbers, however a team stores them.

    One team returns ``(N, 2)`` from ``np.argwhere``; another returns a list of
    ``(int, int)`` tuples. Both are peaks.

    A pair whose first element is itself a tuple is a fingerprint, not a peak:
    a peak is a place in the spectrogram, so both halves are numbers, while a
    fingerprint is ``(key, time)`` and the key is a composite. Two 2026
    repositories make the difference matter. `fingerprint_maker`'s
    ``generate_fingerprints`` returns ``((1, 262, 9), 0)`` and `newton's_code`'s
    ``create_fingerprints`` returns ``((f1, f2, dt), t)``; without this both
    passed the peaks step, so their own fingerprinter bound as the peak finder
    and the search then looked for a third function to fingerprint the
    fingerprints. Measured on rutvim2009 Week1: the chain that came out of that
    fingerprints the log spectrogram instead of the peaks and scores 0.094
    where the peaks score 0.547.
    """

    if isinstance(value, np.ndarray):
        return value.ndim == 2 and value.shape[0] > 0 and value.shape[1] == 2
    try:
        first = next(iter(value))
    except (TypeError, StopIteration):
        return False
    try:
        if len(first) != 2:
            return False
    except TypeError:
        return False
    return not isinstance(first[0], (tuple, list, np.ndarray))


def looks_like_a_whole_spectrogram(value: Any) -> bool:
    """A bare two-dimensional array of more than two columns.

    Narrower than `looks_like_spectrogram`, which reads into a tuple to find
    an array. This is the thing itself, and it exists for one job: to say that
    a spectrogram must not be handed to the fingerprint step. See the
    fingerprints stage below.
    """

    return (
        isinstance(value, np.ndarray)
        and value.ndim == 2
        and value.shape[0] > 1
        and value.shape[1] > 2
    )


def looks_like_fingerprints(value: Any) -> bool:
    """Pairs of (key, time) with a hashable key, or a mapping of one to the other.

    The key is what goes in the database, so it has to be hashable; that single
    property separates a fingerprint list from a peak list without caring
    whether the key is a tuple, an int, or a string.

    A mapping of key to time is the same answer written down differently, and
    one 2026 repository writes it that way: `Cog-gurts`'s
    ``fingerprint_recording`` returns ``{(f1, f2, dt): t}``. Iterating a dict
    yields its keys, so such a value used to pass only by accident, when the
    key happened to be a triple, and a two-element key made it pass for the
    wrong reason. Reading the mapping as a mapping says what is true.

    A flat triple is not a fingerprint here. Every Week 1 fingerprinter in the
    ground truth returns the same two things and nothing else, four of them as
    ``((f1, f2, dt), t)`` pairs and the fifth as a ``{(f1, f2, dt): t}``
    mapping, so nothing in the corpus says which field of a bare ``(a, b, c)``
    is the key and which is the time. Accepting the shape anyway was a rule
    with no case behind it and a consumer that had to guess: `_fingerprint_items`
    read the first two fields, so a team whose rows were ``(f1, f2, dt)`` with
    the time nowhere in them would have had every fingerprint stored under
    ``f1`` at offset ``f2``, and the store would not have raised. A shape the
    benchmark cannot read a key and a time out of is refused at this step,
    where the refusal names the step, rather than passed on and corrupted.
    """

    if isinstance(value, dict):
        if not value:
            return False
        try:
            hash(next(iter(value)))
        except TypeError:
            return False
        return True
    try:
        first = next(iter(value))
    except (TypeError, StopIteration):
        return False
    if not isinstance(first, (tuple, list)) or len(first) != 2:
        return False
    try:
        hash(first[0])
    except TypeError:
        return False
    return True


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
            # One 2026 team wrote `identifying_peaks(samples, rate)`, which
            # computes the spectrogram and finds its peaks in one function.
            # The course names two steps; they wrote one, and it does both.
            # Insisting on a separate spectrogram would refuse a complete
            # working pipeline over how it was divided into functions.
            fusible=True,
        ),
        Stage(
            "fingerprints",
            prefers=("fingerprint", "fgp", "fanout", "hash", "pair"),
            produces=looks_like_fingerprints,
            # The one thing this step must not be handed is the spectrogram.
            # When the step before it returns `(spectrogram, peaks)`, the
            # search offers the whole tuple, then each element in order, and
            # takes the first that works -- so element 0, the spectrogram,
            # gets there first. `fingerprint_maker.generate_fingerprints`
            # accepts it, reads each row's first two numbers as a peak, and
            # returns 2970 fingerprints that pass every shape check and pass
            # the acceptance test on the fixture. Only the scored run tells
            # them apart: 0.094 against 0.547 for the peaks (rutvim2009
            # Week1). Everything except a bare wide array is still offered,
            # so this refuses one thing rather than requiring one thing.
            accepts=lambda value: not looks_like_a_whole_spectrogram(value),
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


#: Fingerprints for the fixture, keyed by the bound call that produced them.
#: One repository tries 3962 store-and-query pairings, and the chain is already
#: bound by then: the same two songs and the same clip go through the same
#: three functions every time, for the same answer. Running them once took a
#: 278-second check down to 21. Cleared per resolution because holding a
#: student's arrays after their run is memory nobody asked us to keep.
_FINGERPRINTED: dict = {}

#: The clip is cut from this song, so this is the id a correct binding names.
TARGET = "fixture_a"


def _binding_key(chain: Sequence[Any]):
    """What makes two chains the same run of the same code.

    Every field `Candidate.bound` reads, because every one of them changes
    what the call is: the tuning it carries, which form of the input it took,
    its argument plan and keywords, whether it runs once per item, which
    element of each result is its output, whether it answers in place, and
    which reading of the upstream value it was handed. The same fields, in
    the same order, as the key `cogbench.pipeline` already uses to ask
    whether it has tried a chain.

    Labels alone are not that. Two bound calls can share every label and
    differ in all of the above -- a method taken off a second object of the
    same class shares its label with the first -- and the second would then
    have been handed the first one's fingerprints without ever running. So
    the callable is identified by address as well, and the chain is kept
    beside its answer (see `_fingerprinted`) so no address can be freed and
    reused while the entry that names it is still readable.

    The tuning is compared by address and by `repr`. Two distinct tunings
    cannot then be read as one; two spellings of the same one at worst cost a
    recompute, which is the direction to be wrong in.
    """

    return tuple(
        (
            id(step.call),
            step.label,
            step.plan,
            step.keywords,
            id(step.tuning),
            repr(step.tuning),
            step.form,
            step.per_item,
            step.element,
            step.self_only,
            step.in_place,
            step.handoff,
        )
        for step in chain
    )


def _fingerprinted(chain: Sequence[Any], songs, sample_rate: int):
    """Fingerprints for both fixture songs and the query clip, computed once.

    Their fingerprinting is pure with respect to the search: it turns audio
    into fingerprints and the search never changes the audio. Whatever it
    writes to disk while doing so happens under `_fresh_state` the first time
    and is not part of what enrolling reads back, because enrolling is what
    gets the fresh directory per attempt.
    """

    key = _binding_key(chain)
    if key not in _FINGERPRINTED:
        with _fresh_state():
            enrolled = [
                (song_id, _run(chain, signal, sample_rate))
                for song_id, signal in songs.items()
            ]
            asked = _run(chain, fixture_clip(songs[TARGET], sample_rate), sample_rate)
        _FINGERPRINTED[key] = (tuple(chain), enrolled, asked)
    _held, enrolled, asked = _FINGERPRINTED[key]
    return enrolled, asked


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
            # The directory this attempt started in can be gone by the time
            # it ends. The search runs inside `pipeline._scratch_cwd`, whose
            # own temporary directory is removed by a garbage-collected
            # finalizer when its generator is abandoned rather than closed;
            # 61 such directories were left behind by one suite run. Raising
            # here would replace whatever this attempt found with a
            # FileNotFoundError from a `finally`, and there is nothing to
            # restore anyway: the point of saving the directory is to leave
            # the caller where it was, and the caller is not there.
            try:
                os.chdir(previous)
            except OSError:
                os.chdir(tempfile.gettempdir())


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
        # Reading the answer runs their code too, when the answer is lazy.
        # `Cog-gurts`'s `fanout_pairs` is a generator function, so a pairing
        # that tries it as the query returns a generator that has not run yet,
        # and the first `next` on it raised ValueError from their own line.
        # Outside this block that exception left `accepts` entirely and
        # aborted the pairing search in `resolve`, which turned one unsuitable
        # candidate into no result for the repository. Six of 2400 enumerated
        # pairings escaped that way.
        best = _winner(answer)
        readable = _scored_ids(answer)
    except BaseException as error:  # noqa: BLE001
        return False, "querying raised {}: {}".format(
            type(error).__name__, str(error)[:120]
        )

    if best != target:
        return False, "asked for {} and got {}".format(
            target, best if best else repr(answer)[:60]
        )

    # Named the right song, so this pairing works. How completely it answered
    # decides which of their own functions the search settles on when several
    # work. One 2026 team wrote `query`, returning the winning song, and
    # `query_details`, returning the same winner plus their whole vote tally.
    # Both are right and the benchmark scores a ranked list, so binding the
    # first one reached cost that team every metric below rank 1.
    #
    # This reads what came back. It never judges their algorithm: a badly
    # tuned fanout should be scored badly by the benchmark, not preferred or
    # refused here.
    #
    # Both grades come off the path scoring uses, so an accepted pairing is
    # one the scorer will read the same way the search did. `_winner` looks
    # deeper into a value than the scorer does: it reaches a song id sitting
    # inside a row the scorer throws the whole list away for, and a pairing
    # graded on that alone is one the benchmark scores at zero. Measured on
    # the 2026 corpus: one team's ranked answer is `[(id, {...}), (id,
    # {...})]`, whose score beside each id is a dict `_as_float` cannot read,
    # and all 68 of its queries were rejected for an identification score of
    # 0.0. Refusing it here is what leaves the search free to keep going,
    # into one of their own readers or another of their queries.
    if len(readable) > 1:
        return 1.0, "identified {} and ranked the rest".format(target)
    if target in readable:
        return 0.5, "identified {} from a two-second clip".format(target)
    return False, "named {} in an answer the benchmark cannot read: {}".format(
        target, repr(answer)[:60]
    )


def _scored_ids(answer: Any) -> list:
    """The song ids the benchmark will read out of their answer, in order.

    Read down the same path scoring uses: `discovered._ranking` unwraps what
    they returned, `checks.coerce_candidates` reads ids out of it, and what
    survives is what the scorer will see. The search is predicting the score,
    so asking the scorer's own two functions is the only way to be sure the
    prediction and the score agree. Every hand-written shape test tried here
    disagreed with them somewhere:

    `[(('fixture_a', 43), 67), (('fixture_a', 44), 11), ...]`, which is what
    rutvim2009 Week1's `get_sorted_matches` ranks, has eight entries and not
    one song id among them. `coerce_candidates` refuses a row whose id is a
    tuple, so grading it complete stopped the search one function short of
    their own `get_sorted_songs`, which turns those buckets into
    `['fixture_a', 'fixture_b']`.

    `[('fixture_a', {...}), ('fixture_b', {...})]`, Asterisk's shape, has two
    entries and a song id in each, and `coerce_candidates` still refuses it:
    the score beside the id is a dict, and `_as_float` cannot read one.

    `('fixture_a', 42)` is a single `(id, score)` pair, which reads as one id
    and not as a two-song ranking, though it is a list of length two with a
    string in front of it.

    The answers that do pass are the ones the scorer can rank: rutvim's
    `['fixture_a', 'fixture_b']`, and carti4ce's `(name, count, tally)`,
    whose tally `_ranking` folds into `[(song, votes), ...]` for the ranks
    below the first.
    """

    from .checks import coerce_candidates
    from .discovered import _ranking

    try:
        return list(coerce_candidates(_ranking(answer)).ids)
    except BaseException:  # noqa: BLE001 - reading a value student code returned
        return []


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
    """Push audio through a resolved chain, changing nothing on the way.

    Every step is called exactly the way the search called it, because every
    part of that call is on the step: the tuning, the input form, the
    per-item loop, and, since `Candidate.handoff`, which reading of the
    previous step's return value this one was handed. `Candidate.bound`
    applies all of them, so the acceptance test, the scored run, and the
    search itself go through one code path.

    This used to re-derive the reading instead, by walking the week's stages
    alongside the chain and offering each element of a tuple until one ran.
    That was a workaround for the reading not being recorded, and it was a
    guess wearing the shape of a rule: rutvim2009 Week1's
    `spectrogram_conversion` returns `(log_spectrogram, peaks)` and their
    `generate_fingerprints` runs on either, so "the first part that runs" is
    the spectrogram, which fingerprints the wrong thing and scores 0.094
    where the peaks score 0.547. The search binds the peaks and now says so.
    """

    steps = list(chain)
    value = steps[0].bound(signal, sample_rate)
    for step in steps[1:]:
        value = step.bound(value)
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
        lambda: _one_at_a_time(store, song_id, fingerprints, order=0),
        lambda: _one_at_a_time(store, song_id, fingerprints, order=1),
    )


#: Where a store that takes one fingerprint per call wants each of the three
#: things it is given, as positions into ``(key, song id, time)``. The first
#: is attested by one repository's ``add_hash(hash_value, song_id, offset)``;
#: the second is the same open question about which comes first that
#: `enroll_arrangements` already answers for a whole song, by trying both.
#:
#: A third order, the offset before the id, is a signature somebody could
#: write and is deliberately not offered. Its cost is not the loop, it is the
#: count: the resolver enumerates every ordered pair of a repository's
#: functions times this many arrangements times the store-and-query shapes,
#: against one ceiling on attempts. Measured on the repository whose database
#: is an argument its own factory makes: 50 candidates, so 2,450 pairs, and
#: the shape with nothing in front of the store costs 2,450 x len(orders) of
#: a 20,000 ceiling. At two orders that is 14,700 and the four shapes that
#: build a database first are still reachable; at three it is 17,150 and they
#: are not, so the search never reaches the pairing that binds and settles
#: for the best thing the first shape offered -- a metadata table filled with
#: songs named after fingerprints, which passes the two-song fixture and
#: scores 0.125 where the real binding scores 0.547. Adding an order costs
#: every repository a seventh of its budget to serve none of them yet.
#:
#: What would make it affordable is not here: the ceiling and the order the
#: shapes are enumerated in belong to the resolver.
_PER_FINGERPRINT_ORDERS = (
    (0, 1, 2),
    (1, 0, 2),
)


#: What one enrolled song was handed to their store as, when it was handed
#: over a fingerprint at a time. Recorded on the binding so the run page can
#: say it, because a store called once per fingerprint is a different call
#: from a store called once per song.
PER_FINGERPRINT = "per fingerprint"

#: How many of their own functions may run on what their query returned before
#: the answer is read. Two, measured: rutvim2009 Week1's `query_database`
#: returns a vote tally keyed by `(song_id, offset)`, `get_sorted_matches`
#: orders it, and `get_sorted_songs` reduces it to song ids. Only the third
#: shape is one the driver can read, and all three functions are theirs.
POST_QUERY_READERS = 2


def _one_at_a_time(store, song_id: str, fingerprints: Any, *, order: int):
    """Offer one fingerprint per call, which is how some teams wrote it.

    `Cog-gurts`'s `AudioDatabase.add_hash(hash_value, song_id, offset)` takes
    a single fingerprint, and the loop over the song's fingerprints is the one
    their own `build_song_database` writes inline. Their store cannot be
    called any other way, and refusing over the loop would refuse a database
    that works.

    The loop is not a step of their algorithm: it decides nothing and computes
    nothing, and the key and the time it passes are exactly the two halves of
    the fingerprint their own fingerprinter produced. Which half goes first is
    the same open question `enroll_arrangements` already answers by trying
    both; ``order`` indexes `_PER_FINGERPRINT_ORDERS`, which says what the
    orders cost and why there are only two.

    A store this cannot be called on is not one of these, and offering it one
    anyway is expensive rather than merely wrong: the loop runs once per
    fingerprint, which is 37,686 calls for one four-second song. `_takes_three`
    is that question and carries what getting it wrong cost.
    """

    if not _takes_three(store):
        raise TypeError("this store does not take a fingerprint, a song id and a time")
    first, second, third = _PER_FINGERPRINT_ORDERS[order]
    for key, when in _fingerprint_items(fingerprints):
        parts = (key, song_id, when)
        store(parts[first], parts[second], parts[third])


def _takes_three(store: Any) -> bool:
    """Whether their store can be called with a fingerprint, an id and a time.

    Two questions, and both have to be answered yes. The signature has to
    bind three positional arguments at all, which is what refuses a store
    that wants two or five. And at least two of the three have to be
    required, which is what refuses a function whose real job takes one
    argument and whose other two parameters are tunings.

    The second is not symmetry. A per-fingerprint store cannot invent the key
    or the name of the song, so those two arrive with every call; the offset
    is the one a store can reasonably default to zero, and requiring all
    three refused such a store over a default it was entitled to write. What
    the second rule keeps out is measured: one repository's
    ``build_song_database(songs_folder, fanout=15, percentile=75)`` requires
    one argument, so a fingerprint key landed in ``songs_folder`` and the two
    halves of the fingerprint in the two tunings. It raised nothing -- it
    globbed a directory and returned an empty database, 75,372 times per
    pairing, at 300 ms an attempt against 15 ms for every other store.

    A function taking ``*args`` is allowed through: it accepts whatever it is
    given by construction, and nothing about its signature says it is not a
    per-fingerprint store. Anything whose signature cannot be read is allowed
    through too, because a builtin or a C-level callable is not evidence
    either way and refusing one would refuse a database that works.
    """

    try:
        signature = inspect.signature(store)
    except (TypeError, ValueError):
        return True
    parameters = list(signature.parameters.values())
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters):
        return True
    try:
        signature.bind(None, None, None)
    except TypeError:
        return False
    required = [
        p
        for p in parameters
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(required) >= 2


def _fingerprint_items(fingerprints: Any):
    """Every fingerprint as ``(key, time)``, whichever way they returned them.

    A mapping is read as a mapping; a sequence of pairs is read as pairs.
    Anything else raises `TypeError`, which the acceptance test already reads
    as "this arrangement does not fit their store" and moves past.

    A row is unpacked rather than indexed, so a row that is not two things
    raises here instead of being cut down to its first two. The two are not
    the same answer: indexing turned a row the benchmark cannot read into a
    key and a time it had no reason to believe, and handed those to their
    store, which stored them without complaint. `looks_like_fingerprints`
    says why nothing in the corpus supports the reading that was being
    guessed at.
    """

    if isinstance(fingerprints, dict):
        return list(fingerprints.items())
    try:
        pairs = [(key, when) for key, when in fingerprints]
    except (TypeError, ValueError) as error:
        raise TypeError(
            "these fingerprints are not a mapping or a sequence of pairs"
        ) from error
    return pairs


def looks_like_an_empty_database(candidate: Any) -> bool:
    """Whether one of their functions hands back a database with nothing in it.

    Week 1's answer to the resolver's question "which of their zero-argument
    functions makes the thing your store and your query both take first". One
    2026 repository writes `create_database()` returning `{}` and then
    `add_fingerprints(db, song_id, fingerprints)` and `query_database(db, fps)`
    (rutvim2009 Week1), so until something makes that dict no pairing of their
    functions can even be tried.

    Empty is the whole test, and it is a property of the value rather than of
    the name: an empty mapping, list or set, or an object carrying no state at
    all. A function that returns a database with songs already in it is not a
    factory, it is a loader, and scoring against it would score their data.

    Every attribute counts, including the ones whose names start with an
    underscore. Skipping those read `self._songs` and `self._hashes` as though
    they were not there, so a loader that keeps its songs private bound as a
    factory and the run was scored against a database that arrived full. The
    cost of the other mistake is much smaller: an object holding some private
    bookkeeping the search cannot size, a lock or a path, is refused as a
    factory and the repository is tried in the shapes that do not need one.
    """

    call = getattr(candidate, "call", candidate)
    try:
        inspect.signature(call).bind()
    except (TypeError, ValueError):
        return False
    # Their factory may write a file next to itself. `_fresh_state` is where
    # every other call the search makes on their code already happens.
    try:
        with _fresh_state():
            made = call()
    except BaseException:  # noqa: BLE001 - student code raises anything
        return False
    if isinstance(made, (dict, list, set, frozenset)):
        return not made
    if made is None or isinstance(made, (str, bytes, tuple, int, float, bool, np.ndarray)):
        return False
    state = _state_of(made)
    if state is None:
        return False
    return not any(_has_contents(value) for value in state)


def _state_of(made: Any):
    """Everything a freshly built object holds, or None if that is unreadable.

    An object keeps its attributes in a ``__dict__``, in slots, or in both,
    and reading only the first refused every database declaring
    ``__slots__``: that object has no ``__dict__`` at all, so a factory that
    returns one looked like an object whose state could not be read and no
    pairing that needed a factory could be tried. A class is free to declare
    slots for the same reason it is free not to, and the question this
    predicate asks -- is the thing empty -- is the same question either way.

    Slots are read through their descriptors rather than by name so the walk
    finds them under the names Python actually stored them under. A slot
    whose name begins with two underscores is stored mangled, and looking it
    up by the name in ``__slots__`` would miss it and read a full database as
    an empty one, which is the expensive direction: see the note above about
    private attributes on `looks_like_an_empty_database`. A slot that was
    never assigned holds nothing and is skipped.

    None means neither kind of state was found, which is not the same as
    finding it empty: a C-level object exposing no attributes at all is not
    evidence that it is a database with nothing in it.
    """

    values: List[Any] = []
    found = False
    attributes = getattr(made, "__dict__", None)
    if isinstance(attributes, dict):
        found = True
        values.extend(attributes.values())
    for owner in type(made).__mro__:
        for member in vars(owner).values():
            if not isinstance(member, MemberDescriptorType):
                continue
            found = True
            try:
                values.append(member.__get__(made, owner))
            except AttributeError:
                continue
    return values if found else None


def _has_contents(value: Any) -> bool:
    """Whether one attribute of a freshly built object already holds data."""

    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return False
    try:
        return len(value) > 0
    except TypeError:
        return True
