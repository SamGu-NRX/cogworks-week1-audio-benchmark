"""Submissions that behave the way real broken student code behaves.

Each of these is a failure the audit saw or a failure the contract has to
survive: raising, returning ``None``, returning a shape nobody documented,
mutating the array we handed over, and taking forever on the first call
because numba is compiling. None of them may end a run.

``ReferenceIdentifier`` is a working fingerprint matcher used only by tests,
never by the scorer. It exists so the ordering assertions (tuned above
detuned above trivial above chance) have something real to order.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from audio_identification_benchmark import synth


def _fanout_hashes(
    signal: np.ndarray, fanout: int, neighborhood: int, percentile: float
) -> List[Tuple[Tuple[int, int, int], int]]:
    """``((f1, f2, dt), t1)`` pairs, the fingerprint the capstone describes."""

    peaks = synth.reference_peaks(
        synth.log_spectrogram(signal), neighborhood=neighborhood, percentile=percentile
    )
    if peaks.shape[0] == 0:
        return []
    order = np.lexsort((peaks[:, 1], peaks[:, 0]))
    peaks = peaks[order]
    frames = peaks[:, 0]
    bins = peaks[:, 1]
    out: List[Tuple[Tuple[int, int, int], int]] = []
    for index in range(peaks.shape[0]):
        stop = min(index + 1 + fanout, peaks.shape[0])
        for other in range(index + 1, stop):
            out.append(
                (
                    (int(bins[index]), int(bins[other]), int(frames[other] - frames[index])),
                    int(frames[index]),
                )
            )
    return out


class ReferenceIdentifier:
    """A working matcher: peaks, fanout hashes, offset-aligned voting.

    ``neighborhood`` and ``percentile`` are the two knobs the course calls
    tunable. The detuned settings in the tests are what a student gets when
    the peak threshold is far too permissive, and the ordering assertion is
    that detuned scores below tuned rather than below any particular number.
    """

    def __init__(
        self,
        resources: Any = None,
        fanout: int = 15,
        neighborhood: int = 15,
        percentile: float = 75.0,
    ) -> None:
        self.fanout = fanout
        self.neighborhood = neighborhood
        self.percentile = percentile
        self._database: Dict[Tuple[int, int, int], List[Tuple[str, int]]] = defaultdict(list)
        self.enroll_counts: Counter = Counter()

    def enroll(self, song_id: str, samples: np.ndarray, sample_rate: int) -> None:
        self.enroll_counts[song_id] += 1
        for key, time in _fanout_hashes(
            samples, self.fanout, self.neighborhood, self.percentile
        ):
            self._database[key].append((song_id, time))

    def identify(self, samples: np.ndarray, sample_rate: int) -> List[Tuple[str, float]]:
        votes: Counter = Counter()
        for key, time in _fanout_hashes(
            samples, self.fanout, self.neighborhood, self.percentile
        ):
            for song_id, stored in self._database.get(key, ()):
                votes[(song_id, stored - time)] += 1
        best: Dict[str, int] = {}
        for (song_id, _offset), count in votes.items():
            if count > best.get(song_id, 0):
                best[song_id] = count
        ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))
        return [(song_id, float(count)) for song_id, count in ranked[:10]]


class BagOfHashesIdentifier(ReferenceIdentifier):
    """Same fingerprints, but votes are summed without aligning offsets.

    Kept because it is the ablation the offset-discipline probe was supposed
    to catch and did not: (f1, f2, dt) hashes already encode local ordering,
    so this scores close to the correct matcher. The tests assert that, so a
    future reader does not rebuild the probe.
    """

    def identify(self, samples: np.ndarray, sample_rate: int) -> List[Tuple[str, float]]:
        votes: Counter = Counter()
        for key, _time in _fanout_hashes(
            samples, self.fanout, self.neighborhood, self.percentile
        ):
            for song_id, _stored in self._database.get(key, ()):
                votes[song_id] += 1
        ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        return [(song_id, float(count)) for song_id, count in ranked[:10]]


class RankedIdsIdentifier(ReferenceIdentifier):
    """Returns ranked ids with no scores: an accepted, score-free shape."""

    def identify(self, samples: np.ndarray, sample_rate: int) -> List[str]:
        return [song_id for song_id, _score in super().identify(samples, sample_rate)]


class SingleAnswerIdentifier(ReferenceIdentifier):
    """Returns one id or ``None``, the shape several audited repos use."""

    def identify(self, samples: np.ndarray, sample_rate: int) -> Optional[str]:
        ranked = super().identify(samples, sample_rate)
        return ranked[0][0] if ranked else None


class TripleIdentifier(ReferenceIdentifier):
    """Returns ``(song_id, artist, score)``, one audited repo's real shape."""

    def identify(self, samples: np.ndarray, sample_rate: int) -> List[Tuple[str, str, float]]:
        return [
            (song_id, "artist for {}".format(song_id), score)
            for song_id, score in super().identify(samples, sample_rate)
        ]


class RaisingIdentifier:
    """Raises on every identify. Ordinary broken code, not an attack."""

    def __init__(self, resources: Any = None) -> None:
        self.enrolled: List[str] = []

    def enroll(self, song_id: str, samples: np.ndarray, sample_rate: int) -> None:
        self.enrolled.append(song_id)

    def identify(self, samples: np.ndarray, sample_rate: int) -> Any:
        raise ValueError("index 4096 is out of bounds for axis 0 with size 2049")


class NoneReturningIdentifier(RaisingIdentifier):
    """Returns ``None`` from identify: a missing return statement."""

    def identify(self, samples: np.ndarray, sample_rate: int) -> Any:
        return None


class WrongShapeIdentifier(RaisingIdentifier):
    """Returns a dict of song to score: plausible, and not an accepted shape."""

    def identify(self, samples: np.ndarray, sample_rate: int) -> Any:
        return {"song-00": 12, "song-01": 3}


class MutatingIdentifier(ReferenceIdentifier):
    """Normalizes the array we handed it, in place.

    This is common (``samples /= np.abs(samples).max()``) and harmless only
    because the driver hands over a fresh copy per call. The test asserts
    the copy, not the etiquette.
    """

    def enroll(self, song_id: str, samples: np.ndarray, sample_rate: int) -> None:
        samples *= 0.0
        super().enroll(song_id, samples, sample_rate)

    def identify(self, samples: np.ndarray, sample_rate: int) -> Any:
        samples[:] = 1.0
        return super().identify(samples, sample_rate)


class EnrollFailsIdentifier(ReferenceIdentifier):
    """Fails to enroll exactly one song, and works for the rest."""

    def __init__(self, resources: Any = None, bad_song: str = "song-00") -> None:
        super().__init__(resources)
        self.bad_song = bad_song

    def enroll(self, song_id: str, samples: np.ndarray, sample_rate: int) -> None:
        if song_id == self.bad_song:
            raise RuntimeError("could not pickle the database for {}".format(song_id))
        super().enroll(song_id, samples, sample_rate)


class ChattyIdentifier(ReferenceIdentifier):
    """Prints on every call. Real, and must not break anything."""

    def identify(self, samples: np.ndarray, sample_rate: int) -> Any:
        for line in range(20):
            print("debug: window {}".format(line))
        return super().identify(samples, sample_rate)
