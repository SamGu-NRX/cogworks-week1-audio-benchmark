"""The ``cogworks.benchmarks.v2`` plugin for the audio-identification benchmark."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .contracts import SAMPLE_RATE, TOP_K, Resources
from .datasets import CacheStatus, load_manifest, materialize_cases, tier_status
from .metrics import score_outputs, trivial_baseline_outcomes
from .sweep import describe as describe_sweep
from .sweep import knee as sweep_knee
from .sweep import sweep_points
from .synth import CORPUS_VERSION


class AudioIdentificationBenchmark:
    """Enroll a synthetic catalog, identify perturbed clips, score the result.

    ``benchmark_id`` is ``audio-identification``, deliberately not the stale
    inactive ``audio-recognition`` row in the seed migration: a metric or
    contract change is a new benchmark, never an in-place edit of one teams
    may already have published against.
    """

    benchmark_id = "audio-identification"
    benchmark_version = 1
    contract_version = "cogworks.submissions.v2"
    plugin_version = __version__
    dataset_version = CORPUS_VERSION
    scorer_version = "identification-v1"
    primary_metric = "identification_score"

    metric_labels = {
        "identification_score": "Identification score",
        "clean_top1": "Clean top-1",
        "noisy_top1": "Noisy top-1",
        "short_clip_top1": "Short clip top-1",
        "pitch_top1": "Pitch-shifted top-1",
        "retrieval_failure_rate": "No candidate found",
        "ranking_failure_rate": "Wrong song ranked first",
        "margin_separation": "Margin separation (AUC)",
        "chance_top1": "Chance",
        "trivial_baseline_top1": "Trivial baseline",
        "median_identify_seconds": "Median identify time",
        "catalog_knee": "Library size at the knee",
    }
    #: Reported only when the curve actually falls. Absent means it held
    #: across the whole sweep, which is the good answer and would read wrong
    #: as a zero.
    conditional_metrics = {"margin_separation", "catalog_knee"}

    #: How the run page labels and reads this benchmark's difficulty sweep.
    #: The axis is in the course's words, since the page draws an axis it
    #: cannot otherwise name; the two keys say which fields of `last_sweep`
    #: are the coordinates.
    sweep_axis_label = "songs in the library"
    sweep_x_key = "catalog_size"
    sweep_y_key = "top1"
    lower_is_better = {
        "retrieval_failure_rate",
        "ranking_failure_rate",
        "median_identify_seconds",
    }

    #: What each metric measures, in the course's own vocabulary, and which
    #: part of the capstone it comes from.
    #:
    #: A benchmark that reports numbers a student cannot trace back to
    #: something they were taught is a black box, and a black box teaches
    #: nothing. Every entry below names the CogWeb section the measurement
    #: corresponds to. Three of the four axes -- clip length, noise level,
    #: catalog size -- are the ones the capstone itself asks students to vary
    #: under "Analyzing and Testing Performance"
    #: (docs/cogweb/pages/Audio/capstone_summary.md). Two more come from the
    #: same page's closing paragraph, which asks how large the leading tally
    #: is against the next-largest, and whether a clip from a song that is not
    #: in the database still returns a match.
    #:
    #: Pitch shift is the exception and is labeled as such: the course never
    #: asks for it. It is here because it is the axis a fingerprint pipeline
    #: measurably fails on, and the failure is worth showing.
    metric_help = {
        "identification_score": (
            "Open-set F1 over every scored clip: precision against how often an "
            "answer was right, recall against how often one was given. Answering "
            "everything collapses precision; never answering zeroes recall. This "
            "is the leaderboard number."
        ),
        "clean_top1": (
            "Studio-quality clips, no perturbation. The capstone's own baseline "
            "case: if this is not near 1.0, the pipeline is broken rather than "
            "merely untuned."
        ),
        "noisy_top1": (
            "Clips with white noise added at a target signal-to-noise ratio -- "
            "the capstone's \"how well does a studio-quality clip, low-noise "
            'clip, ..., or very noisy clip match" question. Noise is added with '
            "signal_power / 10**(snr_db/10), the same formula the course uses."
        ),
        "short_clip_top1": (
            'The capstone\'s "how short of a clip can be matched" question. Fewer '
            "peaks means fewer fanout pairs, so the offset histogram has less to "
            "vote with."
        ),
        "pitch_top1": (
            "Clips shifted in pitch. NOT part of the assignment -- reported "
            "because a shift moves every peak to a different frequency bin, so "
            "the (f1, f2, dt) keys stop matching. Expected behavior of the design "
            "the course teaches, shown rather than hidden."
        ),
        "retrieval_failure_rate": (
            "Share of clips where the right song shared no fingerprint at all "
            "with the query. This is a fingerprinting problem: peak-picking, "
            "fanout, or a sample-rate mismatch between enroll and query."
        ),
        "ranking_failure_rate": (
            "Share of clips where the right song was found but something else won "
            "the tally. This is a voting problem, not a fingerprinting one -- the "
            "capstone's tally step, where votes must be counted per time offset "
            "rather than summed over every hash hit."
        ),
        "margin_separation": (
            'The capstone\'s "how much larger is the leading tally than the '
            'next-largest" question, as an AUC: how well the top-1-versus-top-2 '
            "gap separates correct answers from wrong ones. A high value means "
            "that ratio is a usable confidence signal."
        ),
        "chance_top1": (
            "1/N for a catalog of N songs: what naming a song at random scores. "
            "The floor every other number on this page should be read against."
        ),
        "trivial_baseline_top1": (
            "Whole-clip mean log spectrum, nearest neighbour. No peaks, no "
            "fingerprints, no offsets -- none of the capstone. Beating chance is "
            "not evidence of a working pipeline; beating this is the first sign "
            "that the fingerprinting is doing something."
        ),
        "median_identify_seconds": (
            "Median wall-clock time for one identify call, after warm-up. "
            "Reported, never scored."
        ),
        "catalog_knee": (
            "The library size where identification first drops 15 points below "
            "its best. This is the sweep the course describes: grow the library "
            "and watch where performance degrades. Absent when the curve never "
            "falls that far, which is the good answer."
        ),
    }

    def __init__(self) -> None:
        self.last_diagnostics: List[str] = []
        #: Points for the run page's curve, set by score(). The run page reads
        #: this rather than a metric because a curve is a list, and the metric
        #: table is flat.
        self.last_sweep: List[Dict[str, Any]] = []
        self._baseline_cache: Dict[int, Dict[str, float]] = {}

    def load_cases(self, tier: str, cache_root: Optional[Path] = None) -> Sequence[Any]:
        return materialize_cases(load_manifest(tier))

    def run(self, factory: Any, resources: Any, cases: Sequence[Any]) -> List[Dict[str, Any]]:
        from .drivers import run_cases

        return run_cases(factory, resources, cases)

    def score(self, outputs: Sequence[Dict[str, Any]], cases: Sequence[Any]) -> Dict[str, float]:
        # The trivial baseline is a property of the corpus, not of the
        # submission, so it is computed from the cases and cached per case
        # list rather than recomputed for every scoring call.
        key = id(cases)
        baseline = self._baseline_cache.get(key)
        if baseline is None:
            catalog = {
                case.song_id: case.samples
                for case in cases
                if getattr(case, "kind", "") == "enroll"
            }
            baseline = trivial_baseline_outcomes(cases, catalog)
            self._baseline_cache = {key: baseline}
        metrics, diagnostics = score_outputs(outputs, cases, TOP_K, baseline)

        # How the score moves as the catalog grows, which is the measurement
        # the course actually teaches: "start easy and then start growing the
        # library and see how your performance degrades." Free, because a
        # fingerprint matcher scores each song independently, so restricting
        # the ranking the submission already returned gives the same answer a
        # smaller enrollment would have. Verified against a real
        # re-enrollment at 10 songs: 80 of 80 queries agree, and the curve's
        # predicted top-1 matched the re-run to four decimal places.
        self.last_sweep = sweep_points(outputs, cases)
        sentence = describe_sweep(self.last_sweep)
        if sentence:
            diagnostics.insert(0, sentence)
        at_knee = sweep_knee(self.last_sweep)
        if at_knee is not None:
            metrics["catalog_knee"] = float(at_knee)

        self.last_diagnostics = diagnostics[:32]
        return metrics

    def cache_status(self, tier: str, cache_root: Optional[Path] = None) -> CacheStatus:
        return tier_status(tier)

    def model_factory(self) -> Resources:
        """The value handed to submission factories (rides the model slot).

        The scratch directory is created here and made the working directory
        by the driver before the factory runs, so a submission that opens a
        relative path writes into this run's own directory.
        """

        scratch = Path(tempfile.mkdtemp(prefix="cogworks-week1-"))
        return Resources(sample_rate=SAMPLE_RATE, scratch_dir=scratch, top_k=TOP_K)

    def model_cache_status(self) -> Dict[str, Any]:
        """Nothing to download: the corpus is generated and hash-verified."""

        return {"ready": True, "path": "", "message": "corpus is generated locally"}

    def submission_from_discovery(self, submission: Any) -> Any:
        """Turn a resolved repository into the object ``run`` expects.

        The pairing between what discovery finds and what this benchmark's
        driver calls is this benchmark's to define, which is why it is a method
        here rather than something the resolver knows how to do.
        """

        from .discovered import build

        return build(submission)

    def discovery(self) -> Any:
        """What to look for in a repository that never packaged itself.

        Zero of the thirteen 2026 capstones have a ``pyproject.toml``, a
        ``setup.py``, or a submission entry point, so asking a team to install
        their own code is asking for a step none of them took. Instead the
        benchmark says what its task is -- what goes into the first stage,
        what counts as having done the job -- and ``cogbench.resolve`` searches
        their repository against that by running their functions.

        Built lazily because it renders two songs and importing a plugin
        should not.
        """

        from cogbench.discovery_spec import DiscoverySpec

        from .roles import FINGERPRINT_ROLE, accepts, enroll_arrangements, fixture_songs

        first = next(iter(fixture_songs(SAMPLE_RATE).values()))
        return DiscoverySpec(
            chain_role=FINGERPRINT_ROLE,
            fixture=(first, SAMPLE_RATE),
            accepts=accepts,
            arrangements=enroll_arrangements,
            hints=("week1", "week 1", "audio", "capstone"),
        )
