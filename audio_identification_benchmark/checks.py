"""Normalization of whatever ``identify`` returned into one ranked list.

This module is the only code in the package that reads a student return
value. Every scoring path downstream sees ``Candidates`` and nothing else,
so an accepted alternative shape can only be accepted once, in one place.

Four shapes are accepted, because all four exist in audited repositories:

``[(song_id, score), ...]``
    The documented shape. Full metrics, including score margins.
``[song_id, ...]``
    Ranked ids with no scores. Everything except the margin metrics.
``song_id`` or ``None``
    A single answer. The ranking taxonomy degrades to top-1 versus
    not-found, because nothing below rank 1 was reported.
``[(song_id, artist, score), ...]``
    One repo's ``identify_clip`` returns this; the id is element 0.

Anything else raises ``CheckFailure`` naming the observed type. That failure
belongs to one query, not to the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

#: Strings student code returns to mean "I found nothing". Treated as the
#: absence of a candidate rather than as a song named "Unknown".
_SENTINELS = frozenset({"", "unknown", "none", "null", "no match", "nomatch", "no_match", "n/a"})


class CheckFailure(RuntimeError):
    """A submission return value did not match any accepted shape."""


@dataclass
class Candidates:
    """One query's normalized answer.

    ``ids`` is best-first and may be empty ("no match"). ``scores`` is
    ``None`` when the submission reported ranks without numbers; metrics
    that need numbers report themselves as not measured rather than as
    zero. ``shape`` names which accepted form was seen, for diagnostics.
    """

    ids: List[str]
    scores: Optional[List[float]]
    shape: str


def normalize_song_id(value: Any) -> Optional[str]:
    """Coerce one candidate id to a string, or ``None`` for "no match".

    Sentinels like ``"Unknown"`` and ``""`` become ``None``: several repos
    return them where the contract says ``None``, and scoring them as a
    song named "Unknown" would silently turn a correct abstention into a
    wrong answer.
    """

    if value is None:
        return None
    if isinstance(value, np.str_):
        value = str(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            raise CheckFailure("A candidate id was bytes that are not valid UTF-8.")
    if not isinstance(value, str):
        raise CheckFailure(
            "A candidate id has type {}; song ids must be strings.".format(type(value).__name__)
        )
    text = value.strip()
    if text.lower() in _SENTINELS:
        return None
    return text


def _as_float(value: Any, where: str, numeric_text_ok: bool = False) -> float:
    if isinstance(value, str) and numeric_text_ok:
        # Only reachable when the whole return value was one numpy array with
        # a string dtype: `np.array([["song-01", 5.0]])` stringifies the score
        # by numpy's own promotion rule before we ever see it, so refusing
        # here would report a mistake numpy made, not one the student made.
        try:
            return float(value)
        except ValueError:
            raise CheckFailure(
                "{} is {!r}, which is not a number.".format(where, value[:40])
            )
    if isinstance(value, (bool, str, bytes)) or value is None:
        raise CheckFailure(
            "{} has type {}; a candidate score must be a number.".format(
                where, type(value).__name__
            )
        )
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise CheckFailure(
            "{} has type {}; a candidate score must be a number.".format(
                where, type(value).__name__
            )
        )
    if not np.isfinite(number):
        raise CheckFailure("{} is {}; candidate scores must be finite.".format(where, number))
    return number


def _is_id_like(value: Any) -> bool:
    return value is None or isinstance(value, (str, bytes, np.str_))


def _row_family(row: Any, index: int) -> str:
    """Which accepted per-row shape ``row`` is, or raise naming its type."""

    if _is_id_like(row):
        return "id"
    if isinstance(row, np.ndarray):
        row = row.tolist()
    if isinstance(row, (list, tuple)):
        if len(row) in (2, 3):
            if _is_id_like(row[0]):
                return "pair" if len(row) == 2 else "triple"
            # Right arity, wrong first element. Naming the type of element 0
            # is the difference between "fix your tuple" and "fix your ids".
            raise CheckFailure(
                "identify returned a {}-element {} at position {} whose first element is "
                "{}; the song id must come first and must be a string.".format(
                    len(row), type(row).__name__, index, type(row[0]).__name__
                )
            )
        raise CheckFailure(
            "identify returned a {}-element {} at position {}; expected (song_id, score), "
            "(song_id, artist, score), or a bare song id.".format(
                len(row), type(row).__name__, index
            )
        )
    raise CheckFailure(
        "identify returned {} at position {}; expected (song_id, score), "
        "(song_id, artist, score), or a bare song id.".format(type(row).__name__, index)
    )


ACCEPTED_SHAPES = (
    "a list of (song_id, score) pairs, best first",
    "a list of song ids, best first",
    "a single song id, or None for no match",
    "a list of (song_id, artist, score) triples",
)


def _refuse(value: Any, name: str) -> "CheckFailure":
    return CheckFailure(
        "{} returned {}, which is not an accepted result shape. Return one of: {}.".format(
            name, type(value).__name__, "; ".join(ACCEPTED_SHAPES)
        )
    )


def coerce_candidates(value: Any, name: str = "identify") -> Candidates:
    """Normalize one ``identify`` return value into ranked ids and scores."""

    if isinstance(value, Candidates):
        return value
    # A single string-dtype ndarray stringifies its scores on construction;
    # remembered here so ``_as_float`` can parse them back rather than
    # blaming the student for numpy's promotion.
    numeric_text_ok = False
    if isinstance(value, np.ndarray):
        numeric_text_ok = value.dtype.kind in ("U", "S", "O")
        value = value.tolist()

    # A single answer: "song-03", None, or a sentinel meaning no match.
    if _is_id_like(value):
        song_id = normalize_song_id(value)
        return Candidates([] if song_id is None else [song_id], None, "single_id")

    if isinstance(value, (set, frozenset, dict)):
        # Unordered containers cannot express "best first"; ranking is the
        # whole point, so this is refused rather than silently ordered.
        raise _refuse(value, name)
    if not isinstance(value, (list, tuple)):
        raise _refuse(value, name)

    rows = list(value)
    if not rows:
        return Candidates([], [], "ranked_pairs")

    families = {_row_family(row, index) for index, row in enumerate(rows)}
    if len(families) != 1:
        raise CheckFailure(
            "{} mixed result shapes in one list ({}); every element must have the "
            "same form.".format(name, ", ".join(sorted(families)))
        )
    family = families.pop()

    ids: List[str] = []
    scores: List[float] = []
    for index, row in enumerate(rows):
        if isinstance(row, np.ndarray):
            row = row.tolist()
        if family == "id":
            song_id = normalize_song_id(row)
            if song_id is not None:
                ids.append(song_id)
            continue
        song_id = normalize_song_id(row[0])
        score_index = 1 if family == "pair" else 2
        score = _as_float(
            row[score_index], "{} candidate {} score".format(name, index), numeric_text_ok
        )
        if song_id is None:
            continue
        ids.append(song_id)
        scores.append(score)

    if family == "id":
        return Candidates(ids, None, "ranked_ids")
    return Candidates(ids, scores, "ranked_pairs" if family == "pair" else "ranked_triples")
