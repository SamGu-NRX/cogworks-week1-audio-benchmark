"""Render the corpora and write the manifests that pin them.

Staff tooling. Run it after any change to the synthesis math in
``synth.py``; the sha256 of every rendered song lives in the manifest and
``materialize_cases`` checks it on every run, so a changed generator with
stale manifests is a loud failure rather than a silently different corpus.

    python tools/build_manifests.py

Sizes: the "test" tier is 8 songs of 12 seconds (fast enough to run in a
smoke check), and "evaluation" is 30 songs of 45 seconds. The two tiers draw
from disjoint seed spaces and ``assert_disjoint`` enforces it here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_identification_benchmark.datasets import (  # noqa: E402
    assert_disjoint,
    build_manifest,
)

OUT = ROOT / "audio_identification_benchmark" / "manifests"

#: Distinct master seeds keep the two tiers' song spaces disjoint. A
#: submission tuned against the practice tier must not carry a song it has
#: already seen into the scored tier.
TEST_SEED = 20260817
EVALUATION_SEED = 719241703


def main() -> int:
    test = build_manifest(
        manifest_id="week1-test-v1",
        master_seed=TEST_SEED,
        song_count=8,
        out_of_set_count=2,
        duration_seconds=12.0,
        long_clip=6.0,
        short_clip=2.0,
    )
    evaluation = build_manifest(
        manifest_id="week1-evaluation-v1",
        master_seed=EVALUATION_SEED,
        song_count=30,
        out_of_set_count=6,
        duration_seconds=45.0,
        long_clip=10.0,
        short_clip=3.0,
    )
    assert_disjoint([test, evaluation])

    OUT.mkdir(parents=True, exist_ok=True)
    for name, manifest in (("public-test.json", test), ("public-evaluation.json", evaluation)):
        path = OUT / name
        with open(str(path), "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=1, sort_keys=True)
            stream.write("\n")
        print(
            "{}: {} songs, {} unseen, {} queries, {} bytes".format(
                name,
                len(manifest["songs"]),
                len(manifest["out_of_set_songs"]),
                len(manifest["queries"]),
                path.stat().st_size,
            )
        )
        print("  stats: {}".format(manifest["corpus_stats"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
