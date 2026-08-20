"""Turn a discovered binding into the submission the driver already runs.

Discovery finds which of a team's functions fingerprint a song, store one, and
name one back. The driver wants an object with ``enroll(song_id, samples,
rate)`` and ``identify(samples, rate)``. This is the short piece between them,
and it lives here rather than in ``cogbench`` because the protocol it is
writing to is Week 1's, not the resolver's.

Two rules. Their functions are called exactly as the search called them, with
what the search passed, because the search is what proved the arrangement
works and any deviation here would score a different program than the one that
passed. And nothing is repaired: a function that raises raises, and the driver
records it against the case, which is how a bug in their code stays visible as
a bug in their code.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from .contracts import SAMPLE_RATE

__all__ = ["DiscoveredSubmission", "build"]


class DiscoveredSubmission:
    """A team's own functions, wearing the interface the driver expects."""

    def __init__(self, chain: Sequence[Any], enroll_call, query_call) -> None:
        self._chain = list(chain)
        self._enroll = enroll_call
        self._query = query_call

    def enroll(self, song_id: str, samples, sample_rate: int = SAMPLE_RATE) -> None:
        self._enroll(song_id, self._fingerprint(samples, sample_rate))

    def identify(self, samples, sample_rate: int = SAMPLE_RATE):
        return _ranking(self._query(self._fingerprint(samples, sample_rate)))

    def _fingerprint(self, samples, sample_rate: int):
        """Push audio through their chain, changing nothing on the way."""

        from .roles import _run  # the same call the acceptance test made

        return _run(self._chain, samples, sample_rate)


def build(submission) -> DiscoveredSubmission:
    """Wrap a resolved ``cogbench.resolve.Submission``."""

    if not getattr(submission, "ready", False):
        raise RuntimeError("This repository did not resolve, so there is nothing to run.")
    # Against an empty database. Proving the binding works enrolled two
    # fixture songs, and when a team's database is an object those songs are
    # still in it: `fixture_a` came back in the ranked results for real
    # queries and cost one team half its score.
    ready = submission.fresh()
    return DiscoveredSubmission(ready.chain, ready.enroll, ready.query)


def _ranking(answer: Any) -> Any:
    """Read a ranked list of song ids out of whatever they returned.

    ``checks.coerce_candidates`` reads a list, a list of pairs, or a single id.
    Teams answer in three other shapes, and each is unwrapped by reading what
    is there rather than by rewriting it:

    A dict. One repository returns ``{"best_matches": ..., "ranked": [...],
    "offsets": ...}``, so the value holding a ranking is taken, which is what
    the acceptance test did when it decided this binding was right.

    A tuple whose first element is the answer. Another returns ``(name, count,
    votes)`` from ``query_details``.

    A tally of ``(song_id, offset) -> votes``, which is how both of those
    carry the ranking. Each song's best offset bucket is its score, and the
    songs are ordered by it. Rank 1 is unchanged by this fold: the song owning
    the single largest cell is also the first of the folded order, which is
    what their own ``max(counts, key=counts.get)`` returns. Everything below
    rank 1 is what the fold adds, and it is theirs, not ours: the votes are
    the ones their matcher counted.
    """

    from .roles import looks_like_ranking

    if isinstance(answer, dict):
        folded = _fold_votes(answer)
        if folded is not None:
            return folded
        for value in answer.values():
            if looks_like_ranking(value) or isinstance(value, dict):
                unwrapped = _ranking(value)
                if unwrapped:
                    return unwrapped
        return []
    if isinstance(answer, tuple):
        # Their own order: the answer first, the evidence after it. The
        # richest element wins, so the tally is preferred over the bare name
        # it agrees with.
        for element in reversed(answer):
            if isinstance(element, dict):
                folded = _fold_votes(element)
                if folded is not None:
                    return folded
        for element in answer:
            if looks_like_ranking(element):
                return _ranking(element)
        return []
    return answer


def _fold_votes(tally: Any) -> Any:
    """Turn ``(song_id, offset) -> votes`` into songs, best first.

    Returns None when the mapping is not that shape, so an ordinary dict of
    results falls through to being read as one.
    """

    best: dict = {}
    for key, votes in tally.items():
        if not (isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str)):
            return None
        try:
            count = float(votes)
        except (TypeError, ValueError):
            return None
        song_id = key[0]
        if count > best.get(song_id, float("-inf")):
            best[song_id] = count
    if not best:
        return []
    return [
        (song_id, score)
        for song_id, score in sorted(best.items(), key=lambda row: -row[1])
    ]
