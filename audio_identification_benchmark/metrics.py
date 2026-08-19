"""Controller-side scoring. Pure numpy, no reference fingerprinter.

Every number here is computed from the ranked lists a submission returned.
Nothing in this path runs a working matcher, so an inert or broken
submission lands at retrieval failure and chance rather than at some
accidental constant that happens to look like a score.

The per-query outcome taxonomy is the centre of the design, adopted from an
audited student evaluation harness because it survives contact with real
failures:

``top_1``
    gold is the first candidate.
``top_k``
    gold is in the list but not first.
``ranking_failure``
    gold is in the list but below rank k: the fingerprints matched and the
    vote tally put something else on top.
``retrieval_failure``
    gold is absent entirely: no shared fingerprints at all.

That last split is what makes a mystery result readable. A sample-rate
mismatch and a spectrogram that never resamples both present as near-total
retrieval failure, and the run page can say so in one sentence instead of
showing a number near chance with no reason.

Two baselines are reported next to the primary metric, because "beats
chance" is not evidence of a working pipeline. ``chance_top1`` is 1/N.
``trivial_baseline_top1`` is a whole-clip mean-log-spectrum nearest-neighbour
matcher with no peaks, no fingerprints, and no offsets. On this synthetic
corpus it has not been shown to be a strong floor: measured at 0.242 on the
30-song evaluation tier against a chance baseline of 0.033. Earlier figures
of 0.222 and 0.700 came from a different corpus and a different
implementation and do not describe this code; do not quote them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contracts import TOP_K
from .datasets import QueryCase
from .synth import mean_log_spectrum

#: Outcome names, in the order a run page should show them.
OUTCOMES = ("top_1", "top_k", "ranking_failure", "retrieval_failure", "error", "enrollment_failure")

#: Metrics a submission's own return shape cannot support are omitted from
#: the metric dict entirely and explained in a diagnostic, rather than
#: reported as zero or as NaN. Zero would say the team failed at something
#: they were never asked for; NaN does not survive JSON on the way to the
#: run page, so a column that promised "not measured" would arrive broken.
UNMEASURABLE = ("margin_separation",)


def classify_outcome(
    candidates: Sequence[str], gold: Optional[str], top_k: int = TOP_K
) -> str:
    """The four-way taxonomy for one in-set query."""

    if gold is None:
        raise ValueError("classify_outcome is for in-set queries; gold must be a song id.")
    ids = list(candidates)
    if gold not in ids:
        return "retrieval_failure"
    position = ids.index(gold)
    if position == 0:
        return "top_1"
    if position < top_k:
        return "top_k"
    return "ranking_failure"


def _cell_of(case: QueryCase) -> str:
    """A stable label for the grid cell a query belongs to."""

    if case.kind == "out_of_set":
        return "out_of_set"
    if case.pitch_semitones:
        return "pitch_{:+g}".format(case.pitch_semitones)
    if case.snr_db is not None:
        return "snr_{:g}".format(case.snr_db)
    return "clean_{:g}s".format(case.clip_seconds)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def margin_auc(
    in_set_margins: Sequence[float], out_margins: Sequence[float]
) -> Optional[float]:
    """AUC of a submission's own top1/top2 margin separating in-set from out.

    Computed by rank (the Mann-Whitney identity), so only the ordering of
    the submission's numbers matters and no threshold crosses repositories.
    0.5 means the margin carries no information about whether the song was
    in the database; 1.0 means a single cut on their own numbers would
    separate perfectly.

    Reported as its own labeled column and deliberately kept out of the
    primary metric: most Week 1 pipelines have no abstain path at all (an
    unseen 700 Hz tone still comes back with a confident candidate), and
    folding this in would score teams on work nobody assigned them.
    """

    positive = np.asarray(list(in_set_margins), dtype=np.float64)
    negative = np.asarray(list(out_margins), dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        return None
    combined = np.concatenate([positive, negative])
    order = np.argsort(np.argsort(combined, kind="mergesort"), kind="mergesort") + 1.0
    # Average ranks over ties, otherwise a submission returning one constant
    # margin would score 1.0 or 0.0 depending on concatenation order.
    unique, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=order, minlength=unique.size)
    order = (sums / counts)[inverse]
    rank_sum = float(np.sum(order[: positive.size]))
    return float(
        (rank_sum - positive.size * (positive.size + 1) / 2.0)
        / (positive.size * negative.size)
    )


def _margin(scores: Optional[Sequence[float]]) -> Optional[float]:
    """Relative top1-to-top2 margin from the submission's own numbers.

    Relative, because absolute vote counts are not comparable between two
    repositories or even between two clip lengths in one repository. A list
    with a single candidate has nothing behind it, which is maximal
    separation by the submission's own reckoning.
    """

    if scores is None or len(scores) == 0:
        return None
    if len(scores) == 1:
        return 1.0
    top, second = float(scores[0]), float(scores[1])
    if not np.isfinite(top) or not np.isfinite(second):
        return None
    scale = max(abs(top), 1e-12)
    return float((top - second) / scale)


def trivial_baseline_outcomes(
    cases: Sequence[Any], catalog: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """Score a whole-clip mean-log-spectrum nearest-neighbour matcher.

    Deliberately stupid: one averaged spectrum per song, one per query,
    cosine nearest neighbour. It has no peaks, no fingerprints and no
    temporal information whatsoever, which is what makes it the honest floor
    to print beside chance. Only in-set queries count, matching the primary.
    """

    if not catalog:
        return {"trivial_baseline_top1": 0.0}
    ids = sorted(catalog)
    reference = np.stack([mean_log_spectrum(catalog[song_id]) for song_id in ids])
    reference_center = reference.mean(axis=0)
    reference = reference - reference_center
    norms = np.linalg.norm(reference, axis=1, keepdims=True)
    reference = reference / np.where(norms == 0.0, 1.0, norms)

    hits: List[float] = []
    for case in cases:
        if not isinstance(case, QueryCase) or case.kind != "in_set":
            continue
        # Centre the query by the same catalog per-bin mean used for the
        # reference matrix above. Subtracting the query's own scalar mean
        # instead put the two sides in different spaces, and the nearest
        # neighbour then collapsed onto one song for every query: the floor
        # read as exactly 1/N, which looks like a plausible "the floor is at
        # chance" result and is why the bug survived a calibration run.
        vector = mean_log_spectrum(case.samples) - reference_center
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            hits.append(0.0)
            continue
        similarity = reference @ (vector / norm)
        hits.append(1.0 if ids[int(np.argmax(similarity))] == case.gold_song_id else 0.0)
    return {"trivial_baseline_top1": _mean(hits)}


def score_outputs(
    outputs: Sequence[Dict[str, Any]],
    cases: Sequence[Any],
    top_k: int = TOP_K,
    baseline: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """Every metric plus student-readable diagnostics."""

    diagnostics: List[str] = []
    outcomes: List[str] = []
    per_cell: Dict[str, List[float]] = {}
    seconds: List[float] = []
    in_margins: List[float] = []
    out_margins: List[float] = []
    shapes: Dict[str, int] = {}
    errors: Dict[str, int] = {}
    out_of_set_total = 0
    out_of_set_confident = 0
    enroll_failures: List[str] = []
    # The driver's mapping log and its warm-up note ride on the first
    # output. They are read here rather than in the plugin so a note cannot
    # be computed by the driver and then dropped by whichever caller does
    # the scoring; the same class of loss has bitten this platform before.
    adapter_notes: List[str] = []
    for output in outputs:
        for note in output.get("mappings", []):
            if note not in adapter_notes:
                adapter_notes.append(str(note))
    catalog_ids = {
        case.song_id for case in cases if getattr(case, "kind", "") == "enroll"
    }

    for case, output in zip(cases, outputs):
        kind = getattr(case, "kind", "?")
        if kind == "enroll":
            if not output.get("ok"):
                enroll_failures.append(
                    "{}: {}".format(getattr(case, "song_id", "?"), output.get("error", "?"))
                )
            continue
        if not isinstance(case, QueryCase):
            continue

        cell = _cell_of(case)
        if not output.get("ok"):
            outcome = str(output.get("outcome") or "error")
            outcomes.append(outcome)
            message = str(output.get("error", "no output"))
            errors[message] = errors.get(message, 0) + 1
            if case.kind == "in_set":
                per_cell.setdefault(cell, []).append(0.0)
            else:
                out_of_set_total += 1
            continue

        shapes[str(output.get("shape", "?"))] = shapes.get(str(output.get("shape", "?")), 0) + 1
        elapsed = output.get("seconds")
        if isinstance(elapsed, (int, float)):
            seconds.append(float(elapsed))
        candidates = [str(value) for value in output.get("candidates", [])]
        margin = _margin(output.get("scores"))

        if case.kind == "out_of_set":
            out_of_set_total += 1
            if candidates:
                out_of_set_confident += 1
            if margin is not None:
                out_margins.append(margin)
            continue

        outcome = classify_outcome(candidates, case.gold_song_id, top_k)
        outcomes.append(outcome)
        per_cell.setdefault(cell, []).append(1.0 if outcome == "top_1" else 0.0)
        if margin is not None:
            in_margins.append(margin)

    scored = len(outcomes)
    counts = {name: outcomes.count(name) for name in OUTCOMES}
    identification_score = _mean(
        [value for values in per_cell.values() for value in values]
    )

    pitch_cells = [key for key in per_cell if key.startswith("pitch_")]
    clean_cells = [key for key in per_cell if key.startswith("clean_")]
    snr_cells = [key for key in per_cell if key.startswith("snr_")]

    clean_top1 = _mean(per_cell.get(_longest_clean(clean_cells, per_cell), []))
    short_top1 = _mean(per_cell.get(_shortest_clean(clean_cells), []))
    noisy_top1 = _mean(per_cell.get(_worst_snr(snr_cells), []))
    pitch_top1 = _mean([value for key in pitch_cells for value in per_cell[key]])

    metrics: Dict[str, float] = {
        "identification_score": identification_score,
        "clean_top1": clean_top1,
        "noisy_top1": noisy_top1,
        "short_clip_top1": short_top1,
        "pitch_top1": pitch_top1,
        "retrieval_failure_rate": (counts["retrieval_failure"] / scored) if scored else 0.0,
        "ranking_failure_rate": (counts["ranking_failure"] / scored) if scored else 0.0,
        "chance_top1": (1.0 / len(catalog_ids)) if catalog_ids else 0.0,
        "trivial_baseline_top1": float((baseline or {}).get("trivial_baseline_top1", 0.0)),
        "median_identify_seconds": float(np.median(seconds)) if seconds else 0.0,
    }
    separation = margin_auc(in_margins, out_margins)
    if separation is not None:
        metrics["margin_separation"] = separation

    diagnostics.extend(
        _diagnostics(
            counts,
            scored,
            metrics,
            shapes,
            errors,
            enroll_failures,
            out_of_set_total,
            out_of_set_confident,
            per_cell,
        )
    )
    diagnostics.extend(adapter_notes)
    return metrics, diagnostics


def _longest_clean(clean_cells: Sequence[str], per_cell: Dict[str, List[float]]) -> str:
    if not clean_cells:
        return ""
    return max(clean_cells, key=lambda key: _clean_seconds(key))


def _shortest_clean(clean_cells: Sequence[str]) -> str:
    if not clean_cells:
        return ""
    return min(clean_cells, key=lambda key: _clean_seconds(key))


def _clean_seconds(key: str) -> float:
    try:
        return float(key[len("clean_") : -1])
    except ValueError:
        return 0.0


def _worst_snr(snr_cells: Sequence[str]) -> str:
    if not snr_cells:
        return ""
    def value(key: str) -> float:
        try:
            return float(key[len("snr_") :])
        except ValueError:
            return 0.0
    return min(snr_cells, key=value)


def _diagnostics(
    counts: Dict[str, int],
    scored: int,
    metrics: Dict[str, float],
    shapes: Dict[str, int],
    errors: Dict[str, int],
    enroll_failures: Sequence[str],
    out_of_set_total: int,
    out_of_set_confident: int,
    per_cell: Dict[str, List[float]],
) -> List[str]:
    """Specific sentences a student can act on, in priority order."""

    lines: List[str] = []

    if enroll_failures:
        lines.append(
            "{} songs failed to enroll, so every query for them scored zero. First: "
            "{}".format(len(enroll_failures), enroll_failures[0])
        )

    if scored and metrics["retrieval_failure_rate"] > 0.9:
        lines.append(
            "Almost every query came back with no shared fingerprints at all, which "
            "usually means your database and your queries are in different hash spaces. "
            "Check that one code path resamples both, and that the times you store are "
            "the same units you compare."
        )
    elif scored and metrics["retrieval_failure_rate"] > 0.4:
        lines.append(
            "{:.0%} of queries found no candidate at all. Retrieval, not ranking, is "
            "what is failing: the clip's fingerprints are not landing on the same keys "
            "the database stored.".format(metrics["retrieval_failure_rate"])
        )

    if scored and metrics["ranking_failure_rate"] > 0.15:
        # Two different defects land here, and which one it is depends on
        # whether the submission returns every song with at least one hash
        # hit. A small catalog plus a permissive candidate list turns the
        # hash-space mismatch below into ranking failure rather than
        # retrieval failure (measured on the test tier: a database built at
        # 16 kHz and queried at 44.1 kHz kept gold in the list but dropped
        # its votes from a median of 48 to 8), so both causes are named.
        lines.append(
            "{:.0%} of queries had the right song somewhere in the list but not near the "
            "top. Retrieval found something, so this is the tally losing, not the "
            "fingerprints: check that you count matches per time offset rather than "
            "summing every hash hit, and that the same code path resamples both what you "
            "enroll and what you query.".format(metrics["ranking_failure_rate"])
        )

    if metrics["clean_top1"] > 0.5 and metrics["pitch_top1"] < 0.25:
        lines.append(
            "Clean clips identify at {:.0%} but pitch-shifted clips at {:.0%}. A shifted "
            "clip moves every peak to a different frequency bin, so the stored keys stop "
            "matching; this is the expected behavior of the capstone design and is worth "
            "reporting, not hiding.".format(metrics["clean_top1"], metrics["pitch_top1"])
        )

    if metrics["identification_score"] <= metrics["trivial_baseline_top1"]:
        lines.append(
            "The score is at or below the trivial baseline ({:.3f}), which is a "
            "whole-clip average spectrum with no peaks, no fingerprints and no time "
            "information. Beating chance ({:.3f}) is not the bar; beating this "
            "is.".format(metrics["trivial_baseline_top1"], metrics["chance_top1"])
        )

    if out_of_set_total:
        rate = out_of_set_confident / float(out_of_set_total)
        if rate > 0.9:
            lines.append(
                "Every clip from a song that was never enrolled still came back with a "
                "candidate. Nothing in the assignment required an abstain path, so this "
                "does not affect the score, but a minimum tally or a first-to-second "
                "ratio is what would give you one."
            )
        elif rate < 0.1:
            lines.append(
                "Clips from songs that were never enrolled correctly returned no match "
                "in {:.0%} of cases.".format(1.0 - rate)
            )

    if "margin_separation" not in metrics:
        lines.append(
            "Margin separation is not measured: there was no first-to-second margin to "
            "read on both sides (identify returned ranked ids without scores, or no "
            "out-of-database query produced one). Return (song_id, score) pairs to get "
            "this column."
        )

    if "single_id" in shapes:
        lines.append(
            "identify returned a single id rather than a ranked list for {} queries, so "
            "the ranking breakdown can only distinguish a top-1 hit from a miss.".format(
                shapes["single_id"]
            )
        )

    if errors:
        worst = sorted(errors.items(), key=lambda item: -item[1])[0]
        lines.append("{} queries raised: {}".format(worst[1], worst[0]))

    weakest = _weakest_cell(per_cell)
    if weakest:
        lines.append(
            "Weakest cell: {} at {:.0%} top-1.".format(weakest[0], weakest[1])
        )

    lines.append(
        "Outcome split over {} scored queries: {}.".format(
            scored,
            ", ".join(
                "{} {}".format(counts[name], name) for name in OUTCOMES if counts.get(name)
            )
            or "none",
        )
    )
    return lines


def _weakest_cell(per_cell: Dict[str, List[float]]) -> Optional[Tuple[str, float]]:
    scored = {key: _mean(values) for key, values in per_cell.items() if values}
    if not scored:
        return None
    key = min(scored, key=lambda name: scored[name])
    return key, scored[key]
