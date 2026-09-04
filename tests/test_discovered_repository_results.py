"""Pins automatic discovery results for repositories in the optional corpus.

The corpus is not part of this repository, so these tests skip when its local
checkout or the named repository is absent.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / ".cache" / "student-repos"

sys.path.insert(0, str(ROOT / "python" / "cogbench" / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "week1"))


def _score(plugin, factory) -> dict:
    cases = plugin.load_cases("test")
    resources = plugin.model_factory()
    previous = os.getcwd()
    os.chdir(resources.scratch_dir)
    try:
        return plugin.score(plugin.run(factory, resources, cases), cases)
    finally:
        os.chdir(previous)


@unittest.skipUnless(CORPUS.is_dir(), "student corpus is not checked out")
class DiscoveredRepositoryResults(unittest.TestCase):
    def _assert_result(
        self, name: str, expected_labels: list[str], expected_score: float
    ) -> None:
        from audio_identification_benchmark.discovered import build
        from cogbench.plugins import load_benchmark
        from cogbench.resolve import resolve

        repo = (CORPUS / name).resolve()
        if not repo.is_dir():
            self.skipTest("{} is not in this checkout".format(name))

        plugin = load_benchmark("audio-identification")
        spec = plugin.discovery()
        found = resolve(
            repo,
            chain_role=spec.chain_role,
            fixture=spec.fixture,
            accepts=spec.accepts,
            arrangements=spec.arrangements,
            hints=spec.hints,
        )

        self.assertEqual(found.verdict.status, "scored")
        self.assertEqual([step.label for step in found.chain], expected_labels)

        score = _score(plugin, lambda *args, **kwargs: build(found))
        self.assertAlmostEqual(
            score["identification_score"], expected_score, places=4
        )

    def test_carti4ce_week1_capstone(self):
        self._assert_result(
            "carti4ce__week1_capstone",
            [
                "spectrogram.make_spectrogram",
                "fingerprint.find_peaks",
                "fingerprint.make_fgp",
            ],
            0.65625,
        )

    def test_krazeecoder_week1_capstone_team4(self):
        self._assert_result(
            "KrazeeCoder__week1-capstone-team4",
            [
                "create_spectogram.create_spectrogram",
                "find_peaks.find_peaks",
                "create_fingerprints.peaks_to_fingerprints",
            ],
            0.59375,
        )


if __name__ == "__main__":
    unittest.main()
