"""Run the enrollment and query cases through a submission.

The containment rule is that no ordinary broken submission may end the run.
A student who raises, returns ``None``, returns the wrong shape, mutates our
array in place, or hangs on one clip is writing ordinary code on a Tuesday,
and the run has to come back with a scored result and a sentence saying what
happened. Each case is therefore its own failure domain: one query that
raises becomes that query's outcome and the next query still runs, and one
song that fails to enroll marks its own queries rather than the catalog.

Four things happen before any student code is touched, each for a measured
reason:

- ``matplotlib.use("Agg")``, because the course tells students to build
  spectrograms with ``matplotlib.mlab`` and an interactive backend in a
  headless runner is a hang, not an error.
- ``stdin`` is closed, so an ``input()`` left in a notebook-derived module
  raises immediately instead of waiting forever.
- the process working directory becomes a private scratch directory, because
  one audited repo keeps a module-global relative ``db.pkl`` and rewrites it
  on every add and every query; two runs in one directory corrupt each other.
- a warm-up ``identify`` runs on a one-second dummy clip, so a numba
  first-call JIT is attributed to a named warm-up step instead of looking
  like the first query hanging.
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .adapters import adapt_identifier, instantiate
from .checks import CheckFailure, coerce_candidates
from .contracts import AdapterContractError, Resources
from .datasets import EnrollCase, QueryCase

#: How many candidates are kept from each ranked list. Deeper lists cost
#: payload for ranks nobody scores; the taxonomy only distinguishes down to
#: "present but outside top-k".
KEEP_CANDIDATES = 16


def prepare_process(scratch_dir: Optional[Path] = None) -> Path:
    """Make the process safe for student imports; return the scratch dir."""

    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception:  # noqa: BLE001 - matplotlib is optional for the scorer
        pass
    try:
        if sys.stdin is not None and not sys.stdin.closed:
            sys.stdin.close()
    except Exception:  # noqa: BLE001 - already closed, or not a real stream
        pass
    sys.stdin = io.StringIO("")

    if scratch_dir is None:
        import atexit
        import shutil
        import tempfile

        scratch_dir = Path(tempfile.mkdtemp(prefix="cogworks-week1-"))
        # Removed when the process ends, not here: the run reads from it
        # until scoring is over. One directory leaked per run before this.
        atexit.register(shutil.rmtree, scratch_dir, ignore_errors=True)
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(str(scratch_dir))
    return scratch_dir


def fresh_copy(samples: np.ndarray) -> np.ndarray:
    """A writeable, C-contiguous float32 copy, so mutation cannot leak."""

    return np.array(samples, dtype=np.float32, copy=True, order="C")


def _first_line(error: BaseException) -> str:
    text = str(error)
    return text.splitlines()[0][:220] if text else type(error).__name__


def _error_output(case: Any, kind: str, error: BaseException, extra: Sequence[str] = ()) -> Dict[str, Any]:
    lines = [_first_line(error)]
    lines.extend(str(item)[:220] for item in extra)
    output: Dict[str, Any] = {
        "ok": False,
        "kind": kind,
        "error": " | ".join(lines),
    }
    if isinstance(case, QueryCase):
        output["query_id"] = case.query_id
    elif isinstance(case, EnrollCase):
        output["song_id"] = case.song_id
    return output


def error_outputs(cases: Sequence[Any], error: BaseException) -> List[Dict[str, Any]]:
    """Every case failed the same way: the factory produced nothing usable."""

    if isinstance(error, AdapterContractError):
        extra: Sequence[str] = list(error.report)
    else:
        extra = ["the submission factory raised while constructing the adapter"]
    return [
        _error_output(case, getattr(case, "kind", "?"), error, extra) for case in cases
    ]


def _warm_up(adapter: Any, sample_rate: int) -> Optional[str]:
    """One throwaway identify, so JIT and lazy imports are not query 1's cost.

    A failure here is informational only. A submission that cannot answer a
    one-second clip of silence still gets to run every real query, because
    "we asked you something we never scored and you raised" is not a reason
    to zero a run.
    """

    dummy = np.zeros(sample_rate, dtype=np.float32)
    try:
        adapter.identify(dummy, sample_rate)
    except KeyboardInterrupt:
        raise
    # BaseException: this is the FIRST student call in the run, so a
    # sys.exit() here would end the run before any case was scored.
    except BaseException as error:  # noqa: BLE001 - informational only
        return "warm-up identify on a 1s silent clip raised: {}".format(_first_line(error))
    return None


def run_cases(
    factory: Any, resources: Any, cases: Sequence[Any]
) -> List[Dict[str, Any]]:
    """Instantiate, enroll, query. One output per case, in case order."""

    scratch = prepare_process(getattr(resources, "scratch_dir", None))
    if isinstance(resources, Resources) and resources.scratch_dir is None:
        resources.scratch_dir = scratch
    try:
        adapter = adapt_identifier(instantiate(factory, resources))
    except KeyboardInterrupt:
        raise
    # BaseException: a student sys.exit() at construction must produce a
    # scored zero with a report, not end the run.
    except BaseException as error:  # noqa: BLE001 - student code; report, don't crash
        return error_outputs(cases, error)
    return run_with_adapter(adapter, cases, getattr(resources, "sample_rate", 44100))


def run_with_adapter(
    adapter: Any, cases: Sequence[Any], sample_rate: int = 44100
) -> List[Dict[str, Any]]:
    outputs_by_index: Dict[int, Dict[str, Any]] = {}
    enrolled: Dict[str, bool] = {}
    enroll_errors: Dict[str, str] = {}
    notes: List[str] = []

    # --- enrollment, exactly once per song ------------------------------
    # Re-enrolling a song leaves the key count unchanged but doubles its
    # votes (measured: 118 to 236), which would silently advantage whichever
    # song we happened to send twice. The assertion is here rather than in a
    # comment because a future edit to case ordering is exactly how it
    # would happen.
    for index, case in enumerate(cases):
        if not isinstance(case, EnrollCase):
            continue
        if case.song_id in enrolled:
            outputs_by_index[index] = _error_output(
                case,
                "enroll",
                CheckFailure(
                    "song {} appears twice in the enrollment list; the benchmark enrolls "
                    "each song exactly once.".format(case.song_id)
                ),
            )
            continue
        started = time.time()
        try:
            adapter.enroll(case.song_id, fresh_copy(case.samples), case.sample_rate)
            enrolled[case.song_id] = True
            outputs_by_index[index] = {
                "ok": True,
                "kind": "enroll",
                "song_id": case.song_id,
                "seconds": round(time.time() - started, 4),
            }
        except KeyboardInterrupt:
            raise
        # BaseException, not Exception: a student calling sys.exit() inside
        # enroll raises SystemExit, which an Exception handler lets through
        # and which then ends the whole run instead of marking one song.
        except BaseException as error:  # noqa: BLE001 - one song, not the run
            enroll_errors[case.song_id] = _first_line(error)
            outputs_by_index[index] = _error_output(case, "enroll", error)

    if getattr(adapter, "has_finalize", False):
        try:
            adapter.finalize_database()
        except KeyboardInterrupt:
            raise
        except BaseException as error:  # noqa: BLE001 - optional surface
            notes.append("finalize_database raised: {}".format(_first_line(error)))

    warm_up_note = _warm_up(adapter, sample_rate)
    if warm_up_note:
        notes.append(warm_up_note)

    # --- queries, isolated ----------------------------------------------
    for index, case in enumerate(cases):
        if not isinstance(case, QueryCase):
            continue
        gold = case.gold_song_id
        if gold is not None and gold in enroll_errors:
            output = _error_output(
                case,
                "query",
                CheckFailure(
                    "song {} failed to enroll, so this query could not be answered: "
                    "{}".format(gold, enroll_errors[gold])
                ),
            )
            output["outcome"] = "enrollment_failure"
            outputs_by_index[index] = output
            continue
        try:
            samples = fresh_copy(case.samples)
        except Exception as error:  # noqa: BLE001 - our own corpus; loud, not silent
            outputs_by_index[index] = _error_output(case, "query", error)
            continue
        started = time.time()
        try:
            raw = adapter.identify(samples, case.sample_rate)
            elapsed = time.time() - started
            candidates = coerce_candidates(raw, "identify")
            outputs_by_index[index] = {
                "ok": True,
                "kind": "query",
                "query_id": case.query_id,
                "candidates": candidates.ids[:KEEP_CANDIDATES],
                "scores": (
                    None
                    if candidates.scores is None
                    else [float(value) for value in candidates.scores[:KEEP_CANDIDATES]]
                ),
                "shape": candidates.shape,
                "seconds": round(elapsed, 4),
            }
        except (CheckFailure, AdapterContractError) as error:
            outputs_by_index[index] = _error_output(
                case, "query", error, getattr(error, "report", [])
            )
        except KeyboardInterrupt:
            raise
        # BaseException for the same reason as the enroll loop above: a
        # student sys.exit() must cost one query, not the run.
        except BaseException as error:  # noqa: BLE001 - one query, not the run
            outputs_by_index[index] = _error_output(
                case, "query", error, ["identify raised while running this clip"]
            )

    outputs = [
        outputs_by_index.get(index)
        or _error_output(
            case, getattr(case, "kind", "?"), CheckFailure("case was never executed.")
        )
        for index, case in enumerate(cases)
    ]

    mappings = list(getattr(adapter, "mappings", [])) + notes
    provenance = getattr(adapter, "provenance", None)
    if isinstance(provenance, dict) and provenance.get("source") != "student":
        # Rides the same channel as the mapping log so it reaches the run
        # page without a new field. Stated first, because "we wrote some of
        # this" outranks any note about how a name was resolved.
        supplied = provenance.get("we_supplied") or []
        mappings.insert(
            0,
            "scored through an instructor-supplied adapter; we supplied: {}".format(
                "; ".join(str(item) for item in supplied) or "wiring only"
            ),
        )
    if mappings and outputs:
        outputs[0].setdefault("mappings", mappings[:12])
    return outputs
