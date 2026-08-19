"""What a submission provides and what the benchmark hands it.

A student repository exposes one object with exactly two required methods::

    enroll(song_id: str, samples: np.ndarray, sample_rate: int) -> None
    identify(samples: np.ndarray, sample_rate: int) -> List[Tuple[str, float]]

``identify`` returns ranked candidates, best first; ``[]`` means "no match".
Every array the benchmark hands over is a fresh, writeable, C-contiguous
float32 mono signal in ``[-1, 1]`` at 44100 Hz, so in-place mutation by a
submission is safe and cannot contaminate a later query.

Two optional methods unlock extra diagnostics and never affect the score::

    fingerprint(samples, sample_rate) -> Iterable[(hashable_key, int_time)]
    finalize_database() -> None      # called once, after all enroll() calls

``Resources`` is what the factory receives. It carries the sample rate the
benchmark will use and a private scratch directory that is already the
process working directory by the time the factory runs: one audited student
repo keeps a module-global relative ``db.pkl`` and rewrites it on every add
and query, so two runs sharing a directory corrupt each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

#: Everything the benchmark renders, enrolls, and queries is at this rate.
#: A submission that assumes 44100 and ignores its ``sample_rate`` argument
#: is therefore not punished for it here; a submission that resamples to
#: something else internally must do so on both the enroll and the identify
#: path or its hash spaces will not overlap (measured: a 16 kHz database
#: queried at 44.1 kHz collapsed a correct song from 118 votes to 4).
SAMPLE_RATE = 44100

#: How deep a candidate list counts as a "top-k" hit in the outcome
#: taxonomy. Adopted from the student harness the taxonomy comes from.
TOP_K = 5


class AdapterContractError(RuntimeError):
    """A submission object could not be mapped onto the identify protocol.

    ``report`` carries the actionable mapping report (what was received,
    which roles are missing, the closest names found, and the escape hatch).
    """

    def __init__(self, message: str, report: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.report: List[str] = list(report or [])


@dataclass
class Resources:
    """The value handed to a submission factory.

    Attributes
    ----------
    sample_rate
        The rate of every array the benchmark passes in.
    scratch_dir
        A private, writable directory for this run. The driver has already
        made it the process working directory, so relative paths a
        submission opens land here.
    top_k
        How many candidates the benchmark reads from each ranked list.
    """

    sample_rate: int = SAMPLE_RATE
    scratch_dir: Optional[Path] = None
    top_k: int = TOP_K
