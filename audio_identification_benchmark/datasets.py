"""Manifests, corpus materialization, and the cases handed to the driver.

A manifest is a versioned JSON file shipped inside the package. It pins the
seeds every song is rendered from, the sha256 each rendered signal must
have, the exact query grid, and the statistics the corpus measured about
itself. Nothing is sampled at run time, so two runs of the same tier score
the same audio and the same queries.

Cases come out in one list, enrollments first: one ``EnrollCase`` per song
in the catalog, then one ``QueryCase`` per query. The controller expects one
output per case, and this ordering is what lets a failed enrollment mark its
own song without touching the rest of the run.

A ``QueryCase`` holds a reference to its source song and derives its clip on
access rather than storing it. The evaluation catalog is 30 songs of 45
seconds (about 240 MB of float32); materializing every query's audio
alongside it would roughly double that for no benefit, since the driver
copies each array before handing it over anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from . import exactmath

from . import synth
from .contracts import SAMPLE_RATE

TIERS = ("test", "evaluation")


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class CacheStatus:
    ready: bool
    path: Path
    message: str


@dataclass
class EnrollCase:
    """One song to put in the submission's database, exactly once."""

    song_id: str
    samples: np.ndarray
    sample_rate: int
    kind: str = "enroll"


@dataclass
class QueryCase:
    """One clip to identify, plus the perturbation that produced it.

    ``gold_song_id`` is the song the clip was cut from, or ``None`` when the
    clip came from a song that was never enrolled. ``kind`` says which of
    those two it is, so a missing gold is never ambiguous with a stripped
    one.
    """

    query_id: str
    sample_rate: int
    clip_seconds: float
    pitch_semitones: float
    snr_db: Optional[float]
    gold_song_id: Optional[str]
    kind: str  # "in_set" | "out_of_set"
    offset_seconds: float = 0.0
    noise_seed: int = 0
    source_song_id: str = ""
    _source: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def samples(self) -> np.ndarray:
        """The clip, derived deterministically from the source song."""

        if self._source is None:
            raise DatasetError(
                "Query {} has no source audio; materialize_cases was not run.".format(
                    self.query_id
                )
            )
        return synth.perturb(
            self._source,
            self.sample_rate,
            self.clip_seconds,
            self.offset_seconds,
            self.pitch_semitones,
            self.snr_db,
            self.noise_seed,
        )


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------


def load_manifest(tier: str) -> Dict[str, Any]:
    """Read the shipped manifest for ``tier``. No network, no cache."""

    if tier not in TIERS:
        raise ValueError("Tier must be one of {}.".format(", ".join(TIERS)))
    name = "public-{}.json".format(tier)
    from importlib import resources as importlib_resources

    package = "audio_identification_benchmark.manifests"
    try:
        text = importlib_resources.files(package).joinpath(name).read_text(encoding="utf-8")
    except AttributeError:  # Python 3.8
        with importlib_resources.open_text(package, name, encoding="utf-8") as stream:
            text = stream.read()
    manifest = json.loads(text)
    if manifest.get("corpus_version") != synth.CORPUS_VERSION:
        raise DatasetError(
            "Manifest {} was built for corpus {}, but this package renders {}. "
            "Rebuild the manifests; do not score against mismatched audio.".format(
                manifest.get("manifest_id", name),
                manifest.get("corpus_version"),
                synth.CORPUS_VERSION,
            )
        )
    return manifest


def materialize_cases(manifest: Mapping[str, Any], verify: bool = True) -> List[Any]:
    """Render the corpus and build every case, enrollments first.

    Hash verification happens here, before a single line of student code
    runs: a corpus that is not the pinned corpus produces numbers that look
    exactly like numbers that mean something.
    """

    sample_rate = int(manifest["sample_rate"])
    catalog, out_of_set = synth.render_from_manifest(dict(manifest), verify=verify)

    cases: List[Any] = [
        EnrollCase(song_id=song_id, samples=catalog[song_id], sample_rate=sample_rate)
        for song_id in sorted(catalog)
    ]

    for row in manifest["queries"]:
        source_id = str(row["source_song_id"])
        in_set = bool(row["in_set"])
        source = catalog.get(source_id) if in_set else out_of_set.get(source_id)
        if source is None:
            raise DatasetError(
                "Query {} names source song {}, which is not in the {} corpus.".format(
                    row.get("query_id"), source_id, "catalog" if in_set else "out-of-set"
                )
            )
        snr = row.get("snr_db")
        cases.append(
            QueryCase(
                query_id=str(row["query_id"]),
                sample_rate=sample_rate,
                clip_seconds=float(row["clip_seconds"]),
                pitch_semitones=float(row["pitch_semitones"]),
                snr_db=None if snr is None else float(snr),
                gold_song_id=source_id if in_set else None,
                kind="in_set" if in_set else "out_of_set",
                offset_seconds=float(row["offset_seconds"]),
                noise_seed=int(row["noise_seed"]),
                source_song_id=source_id,
                _source=source,
            )
        )
    return cases


def tier_status(tier: str) -> CacheStatus:
    """Whether the tier can run. The corpus is generated, so nothing to fetch."""

    try:
        manifest = load_manifest(tier)
    except (OSError, ValueError, KeyError, DatasetError) as error:
        return CacheStatus(ready=False, path=Path("."), message=str(error))
    return CacheStatus(
        ready=True,
        path=Path("."),
        message="{} songs, {} queries, generated locally (no download)".format(
            len(manifest["songs"]), len(manifest["queries"])
        ),
    )


def assert_disjoint(manifests: Iterable[Mapping[str, Any]]) -> None:
    """No song seed may appear in two manifests, in any role.

    Two tiers that share a seed share a song, which would let a submission
    tuned against the practice tier carry an advantage into the evaluation
    tier that has nothing to do with its pipeline. Out-of-set songs are
    included in the check: an out-of-set song in one tier that is enrolled
    in another is the same leak wearing a different label.
    """

    seen: Dict[int, str] = {}
    for manifest in manifests:
        label = str(manifest.get("manifest_id", "?"))
        rows = list(manifest.get("songs", [])) + list(manifest.get("out_of_set_songs", []))
        for row in rows:
            seed = int(row["seed"])
            previous = seen.get(seed)
            if previous is not None and previous != label:
                raise DatasetError(
                    "Song seed {} appears in both {} and {}; the tiers are not "
                    "disjoint.".format(seed, previous, label)
                )
            seen[seed] = label


# --------------------------------------------------------------------------
# Grid description, shared by the manifest builder and the metrics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GridCell:
    """One (clip length, pitch shift, SNR) combination of the query grid."""

    name: str
    clip_seconds: float
    pitch_semitones: float
    snr_db: Optional[float]


#: The scored grid. Built around pitch shift because that is the only axis
#: anyone has measured a separation on: an audited student sweep of 1,275
#: committed trials over five noise types and five SNRs came back at 0.996
#: overall and 0.97 at the worst SNR, so additive noise is saturated and
#: cannot rank teams. A second audited sweep found clean recall of 97-100%
#: collapsing to 10-25% at one or two semitones. The noise cells stay
#: because a submission that falls over on them has a real defect worth
#: naming, not because they discriminate.
def default_grid(long_clip: float, short_clip: float) -> List[GridCell]:
    return [
        GridCell("clean", long_clip, 0.0, None),
        GridCell("short", short_clip, 0.0, None),
        GridCell("noisy", long_clip, 0.0, 0.0),
        GridCell("noisy_mid", long_clip, 0.0, 10.0),
        GridCell("pitch_down_2", long_clip, -2.0, None),
        GridCell("pitch_down_1", long_clip, -1.0, None),
        GridCell("pitch_up_1", long_clip, 1.0, None),
        GridCell("pitch_up_2", long_clip, 2.0, None),
    ]


def build_manifest(
    manifest_id: str,
    master_seed: int,
    song_count: int,
    out_of_set_count: int,
    duration_seconds: float,
    long_clip: float,
    short_clip: float,
    sample_rate: int = SAMPLE_RATE,
    with_stats: bool = True,
) -> Dict[str, Any]:
    """Render a corpus and emit the manifest that pins it.

    Staff tooling only; this never runs in the scoring path. The sha256 of
    every rendered signal goes into the manifest here, and ``load_manifest``
    plus ``materialize_cases`` check it on every run afterwards.
    """

    songs: List[Dict[str, Any]] = []
    for index in range(song_count):
        seed = synth.song_seed(master_seed, index)
        song_id = "song-{:02d}".format(index)
        signal = synth.render_song(
            synth.SongSpec(song_id, seed, duration_seconds), sample_rate
        )
        songs.append(
            {
                "song_id": song_id,
                "seed": seed,
                "duration_seconds": duration_seconds,
                "sha256": synth.sha256_signal(signal),
            }
        )

    out_rows: List[Dict[str, Any]] = []
    for index in range(out_of_set_count):
        # Offset the index space so an out-of-set song can never collide
        # with a catalog song's seed inside one manifest.
        seed = synth.song_seed(master_seed, 10_000 + index)
        song_id = "unseen-{:02d}".format(index)
        signal = synth.render_song(
            synth.SongSpec(song_id, seed, duration_seconds), sample_rate
        )
        out_rows.append(
            {
                "song_id": song_id,
                "seed": seed,
                "duration_seconds": duration_seconds,
                "sha256": synth.sha256_signal(signal),
            }
        )

    grid = default_grid(long_clip, short_clip)
    rng = np.random.default_rng(int(master_seed) ^ 0x5EED)
    queries: List[Dict[str, Any]] = []
    counter = 0
    # Clip offsets avoid the first and last second: the opening of a
    # generated song is sparser than its middle, and a clip that runs off
    # the end would be scored partly against silence.
    usable = max(duration_seconds - long_clip - 1.0, 0.5)
    for cell in grid:
        for row in songs:
            offset = 1.0 + float(exactmath.uniform(rng, 0.0, usable))
            queries.append(
                {
                    "query_id": "q{:04d}".format(counter),
                    "source_song_id": row["song_id"],
                    "in_set": True,
                    "cell": cell.name,
                    "clip_seconds": cell.clip_seconds,
                    "offset_seconds": round(offset, 4),
                    "pitch_semitones": cell.pitch_semitones,
                    "snr_db": cell.snr_db,
                    "noise_seed": int(rng.integers(1, 2**31 - 1)),
                }
            )
            counter += 1
    # Out-of-set queries are clean and long on purpose: the question they
    # ask is whether a submission can say "not in my database" when the
    # audio is easy, not whether it degrades under perturbation.
    for row in out_rows:
        for _ in range(2):
            offset = 1.0 + float(exactmath.uniform(rng, 0.0, usable))
            queries.append(
                {
                    "query_id": "q{:04d}".format(counter),
                    "source_song_id": row["song_id"],
                    "in_set": False,
                    "cell": "out_of_set",
                    "clip_seconds": long_clip,
                    "offset_seconds": round(offset, 4),
                    "pitch_semitones": 0.0,
                    "snr_db": None,
                    "noise_seed": int(rng.integers(1, 2**31 - 1)),
                }
            )
            counter += 1

    manifest: Dict[str, Any] = {
        "manifest_id": manifest_id,
        "corpus_version": synth.CORPUS_VERSION,
        "sample_rate": sample_rate,
        "master_seed": int(master_seed),
        "songs": songs,
        "out_of_set_songs": out_rows,
        "queries": queries,
    }
    if with_stats:
        catalog = {
            row["song_id"]: synth.render_song(
                synth.SongSpec(row["song_id"], int(row["seed"]), duration_seconds), sample_rate
            )
            for row in songs
        }
        manifest["corpus_stats"] = synth.corpus_stats(catalog)
    return manifest


def query_cases(cases: Sequence[Any]) -> List[QueryCase]:
    return [case for case in cases if isinstance(case, QueryCase)]


def enroll_cases(cases: Sequence[Any]) -> List[EnrollCase]:
    return [case for case in cases if isinstance(case, EnrollCase)]
