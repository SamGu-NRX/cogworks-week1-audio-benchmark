"""Discovery must score a repository the same as a human reading it did.

Two repositories in the 2026 corpus have an instructor-written adapter, made
by a person who read the team's code and wired it up by hand. Those are the
only ground truth available for whether an automatic binding found the right
functions, and "it produced a number" is not the same claim as "it produced
the right number".

These are skipped when the corpus is not checked out, because it is not part
of the repository.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / ".cache" / "student-repos"
ADAPTERS = ROOT / "benchmarks" / "adapters"

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
class DiscoveryMatchesTheHandWrittenAdapter(unittest.TestCase):
    def _compare(self, name: str) -> None:
        from cogbench.plugins import load_benchmark
        from cogbench.resolve import resolve

        repo = (CORPUS / name).resolve()
        adapter = ADAPTERS / name / "submission.py"
        if not repo.is_dir() or not adapter.is_file():
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
        self.assertTrue(found.ready, found.verdict.headline)

        from audio_identification_benchmark.discovered import build

        discovered = _score(plugin, lambda *a, **k: build(found))

        # The sandbox stages the adapter into the repository, so it is
        # imported from there: its siblings are the team's own modules.
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        staged = repo / "_adapter_under_test.py"
        staged.write_bytes(adapter.read_bytes())
        self.addCleanup(lambda: staged.unlink(missing_ok=True))
        module_spec = importlib.util.spec_from_file_location("_adapter_under_test", staged)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        by_hand = _score(plugin, module.create_submission)

        self.assertEqual(
            round(discovered["identification_score"], 4),
            round(by_hand["identification_score"], 4),
        )
        for metric, value in sorted(by_hand.items()):
            if metric == "median_identify_seconds":
                continue  # timing, not a claim about their code
            self.assertAlmostEqual(
                discovered.get(metric, float("nan")), value, places=4, msg=metric
            )

    def test_a_repository_whose_database_is_a_module(self):
        self._compare("carti4ce__week1_capstone")

    def test_a_repository_whose_database_is_a_class(self):
        """This one is the reason `Submission.fresh` exists. Proving a binding
        enrolls two fixture songs, and an object keeps them, so scoring from
        there put fixture_a in the ranked results for real queries."""

        self._compare("KrazeeCoder__week1-capstone-team4")


if __name__ == "__main__":
    unittest.main()
