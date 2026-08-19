# CogWorks Week 1: audio identification benchmark

Scores a Week 1 capstone submission on identifying songs from short clips.
A submission enrolls a catalog, then answers queries cut from those songs
after perturbation (short clips, added noise, pitch shift), plus queries cut
from songs that were never enrolled.

## The contract

```python
enroll(song_id: str, samples: np.ndarray, sample_rate: int) -> None
identify(samples: np.ndarray, sample_rate: int) -> List[Tuple[str, float]]
```

`identify` returns ranked candidates, best first; `[]` means no match. Every
array is a fresh, writeable, C-contiguous float32 mono signal in `[-1, 1]`
at 44100 Hz, so mutating it in place is safe.

Three other return shapes are accepted, normalized in `checks.py`: a bare
list of ids (the margin column is then reported as not measured), a single
id or `None`, and `(song_id, artist, score)` triples. Anything else raises
for that one query, naming the type it saw.

Two optional methods add diagnostics and never affect the score:
`fingerprint(samples, sample_rate)` and `finalize_database()`.

## Metrics

`identification_score` is the primary: mean top-1 accuracy over the scored
grid, in-set queries only. Each query also lands in one of four outcomes,
which is what makes a bad result readable rather than mysterious:

| outcome | meaning |
| --- | --- |
| `top_1` | gold is the first candidate |
| `top_k` | gold is in the list, not first |
| `ranking_failure` | gold is in the list, below rank 5 |
| `retrieval_failure` | gold is absent: no shared fingerprints at all |

`chance_top1` (1/N) and `trivial_baseline_top1` are printed beside the
primary. The trivial baseline is a whole-clip mean-log-spectrum nearest
neighbour with no peaks, no fingerprints, and no time information. Beating
chance is not the bar; beating that is.

The grid is built around pitch shift, because that is the only axis anyone
has measured a separation on. An audited student sweep of 1,275 committed
trials over five noise types and five SNRs came back at 0.996 overall, so
additive noise saturates and cannot rank teams.

## Running it

```
pip install -e .
pytest tests/ -q
```

Rebuild the manifests after any change to the synthesis math in `synth.py`
(the sha256 of every rendered song lives in the manifest and is checked on
every run):

```
python tools/build_manifests.py
```

## Layout

| file | what it does |
| --- | --- |
| `synth.py` | seeded non-stationary corpus generator, perturbations, corpus statistics |
| `datasets.py` | manifests, case materialization, the query grid |
| `checks.py` | the one place a student return value is read |
| `adapters.py` | name resolution with arity and signature guards |
| `drivers.py` | per-case isolation, scratch dir, exactly-once enrollment |
| `metrics.py` | the taxonomy, every metric, the diagnostics |
| `plugins.py` | the `cogworks.benchmarks.v2` entry point |
