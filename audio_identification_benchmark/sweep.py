"""How the score changes as the catalog grows.

The course teaches the measurement this file performs. From the Week 1
material: start with a small library, grow it, and watch where performance
degrades. A single number against a single catalog size does not support that
method, so this reports the curve instead.

The measurement is free. A fingerprint matcher scores each enrolled song
independently: a song's vote count depends on the query and on that song's own
fingerprints, never on which other songs happen to be in the database. So the
ranked list a submission returns for the full catalog already contains the
answer for every subset of it. Restricting that list to a subset gives the same
ranking the submission would have produced had only the subset been enrolled,
and the sweep costs no extra calls into student code.

That equivalence is a property of the algorithm the course teaches, not of
every algorithm. A submission that normalizes scores across the catalog, or
that trains a classifier over the enrolled set, would break it, and then this
curve describes something slightly different from a real re-enrollment: it
answers "how well does your ranking hold up when fewer songs count" rather than
"how well would you do with a smaller database." ``verify_subset_equivalence``
below measures the gap directly on a submission, and the run page says which
question the curve answered.

Nested subsets, deterministic by song id, so the 40-song point contains every
song in the 20-song point. Growing the same library is what the course
describes, and it means a knee in the curve is about size rather than about
which songs got picked.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .contracts import TOP_K
from .datasets import QueryCase

#: Catalog sizes to report, smallest first. Capped at the real catalog size,
#: and any point equal to it is dropped, since the full catalog is the headline
#: number and repeating it in the curve would suggest two measurements where
#: there is one.
DEFAULT_SIZES = (5, 10, 20, 40, 80, 160)

#: Below this many songs a point is noise: with 4 songs, chance alone is 25%,
#: and a curve that starts there tells a team their pipeline is strong when it
#: is guessing. Points smaller than this are dropped rather than reported with
#: a caveat nobody reads.
MIN_CATALOG = 5


def subset_ids(catalog_ids: Sequence[str], size: int) -> List[str]:
    """The first ``size`` song ids in sorted order.

    Sorted rather than sampled: the same catalog always yields the same
    subsets, so two runs of one submission produce the same curve and a team
    comparing this week's run against last week's is comparing like with like.
    """

    return sorted(catalog_ids)[:size]


def restrict(candidates: Sequence[str], keep: Sequence[str]) -> List[str]:
    """The ranking with ids outside ``keep`` removed, order preserved.

    This is the whole trick. Removing a song the submission ranked above the
    gold song promotes the gold song, which is exactly what would have happened
    had that song never been enrolled.
    """

    allowed = set(keep)
    return [str(value) for value in candidates if str(value) in allowed]


def sweep_points(
    outputs: Sequence[Dict[str, Any]],
    cases: Sequence[Any],
    sizes: Sequence[int] = DEFAULT_SIZES,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """Top-1 accuracy at each catalog size, plus the full catalog.

    One dict per point: ``catalog_size``, ``top1``, ``queries``, and
    ``retrieval_failure_rate``. The last one is what separates a curve that
    falls because ranking got harder from one that falls because the
    fingerprints stopped matching at all, and those have different fixes.
    """

    catalog_ids = sorted(
        {case.song_id for case in cases if getattr(case, "kind", "") == "enroll"}
    )
    if not catalog_ids:
        return []

    # Pair each in-set query with what the submission returned for it, once,
    # rather than re-walking both sequences per point.
    answered: List[Tuple[str, List[str]]] = []
    for case, output in zip(cases, outputs):
        if not isinstance(case, QueryCase) or case.kind != "in_set":
            continue
        if not output.get("ok"):
            answered.append((str(case.gold_song_id), []))
            continue
        answered.append(
            (str(case.gold_song_id), [str(value) for value in output.get("candidates", [])])
        )
    if not answered:
        return []

    wanted = [size for size in sorted(set(sizes)) if MIN_CATALOG <= size < len(catalog_ids)]
    wanted.append(len(catalog_ids))

    points: List[Dict[str, Any]] = []
    for size in wanted:
        keep = subset_ids(catalog_ids, size)
        allowed = set(keep)
        hits = 0
        misses = 0
        counted = 0
        for gold, candidates in answered:
            # A query whose own song is outside this subset has no right
            # answer at this size. Scoring it as a miss would make the curve
            # fall for a reason that is about our subsetting rather than about
            # the submission.
            if gold not in allowed:
                continue
            counted += 1
            ranking = restrict(candidates, keep)
            if ranking[:1] == [gold]:
                hits += 1
            if gold not in ranking:
                misses += 1
        if not counted:
            continue
        points.append(
            {
                "catalog_size": size,
                "top1": hits / counted,
                "queries": counted,
                "retrieval_failure_rate": misses / counted,
            }
        )
    return points


def knee(points: Sequence[Dict[str, Any]], drop: float = 0.15) -> Optional[int]:
    """The first catalog size where top-1 falls by ``drop`` from its best.

    ``None`` when the curve never falls that far, which is the answer for a
    submission that holds across the whole sweep and for one that was flat at
    zero throughout. The caller separates those by reading the values.
    """

    if len(points) < 2:
        return None
    best = max(point["top1"] for point in points)
    if best <= 0.0:
        return None
    for point in points:
        if best - point["top1"] >= drop:
            return int(point["catalog_size"])
    return None


def describe(points: Sequence[Dict[str, Any]]) -> Optional[str]:
    """One sentence about the curve, in the course's register, or ``None``.

    Assembled from a fixed set of templates. Nothing here is generated, so the
    sentence can only say something the numbers support.
    """

    if len(points) < 2:
        return None
    first, last = points[0], points[-1]
    best = max(point["top1"] for point in points)
    if best <= 0.0:
        return None

    at_knee = knee(points)
    if at_knee is not None:
        # Which half of the pipeline gave way, read off the failure mode at
        # the point where the curve broke.
        broke = next(point for point in points if point["catalog_size"] == at_knee)
        if broke["retrieval_failure_rate"] > 0.4:
            return (
                "Identification holds to a {}-song library, then falls off, and at "
                "that point most queries find no matching fingerprints at all. That "
                "is the fingerprints rather than the vote."
            ).format(at_knee)
        return (
            "Identification holds to a {}-song library, then falls off. The right "
            "song is still being found, so the vote is what gives way as the "
            "library grows."
        ).format(at_knee)

    if last["top1"] >= first["top1"] - 0.05:
        return (
            "Identification holds steady from {} songs to {}, so nothing in the "
            "pipeline is size-limited over this range."
        ).format(first["catalog_size"], last["catalog_size"])
    return (
        "Identification falls gradually from {:.0%} at {} songs to {:.0%} at {}, "
        "without a single point where it breaks."
    ).format(first["top1"], first["catalog_size"], last["top1"], last["catalog_size"])


def verify_subset_equivalence(
    full_outputs: Sequence[Dict[str, Any]],
    subset_outputs: Sequence[Dict[str, Any]],
    cases: Sequence[Any],
    subset: Sequence[str],
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    """Whether restricting the full ranking matches a real re-enrollment.

    Run a submission twice, once on the whole catalog and once with only
    ``subset`` enrolled, and pass both here. It reports how often the two
    agree on the top-1 answer.

    This exists because the sweep's cheapness rests on an assumption about the
    submission's algorithm, and an assumption that is never checked is a claim.
    A staff tool runs this against the reference implementations; a submission
    that disagrees is not wrong, it is telling us its matcher considers the
    catalog as a whole.
    """

    allowed = set(subset)
    agree = 0
    disagree = 0
    for case, full, actual in zip(cases, full_outputs, subset_outputs):
        if not isinstance(case, QueryCase) or case.kind != "in_set":
            continue
        if str(case.gold_song_id) not in allowed:
            continue
        if not full.get("ok") or not actual.get("ok"):
            continue
        predicted = restrict([str(v) for v in full.get("candidates", [])], subset)[:1]
        observed = [str(v) for v in actual.get("candidates", [])][:1]
        if predicted == observed:
            agree += 1
        else:
            disagree += 1
    total = agree + disagree
    return {
        "queries": total,
        "agreement": (agree / total) if total else 0.0,
        "disagreements": disagree,
    }
