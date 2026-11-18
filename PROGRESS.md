# Progress snapshot: CogPortal-audio-diagnostics (20261118)

The local checkout `~/Programming Projects/CogPortal-audio-diagnostics` was removed on 2026-11-18 to free disk space on Sam's Mac, after everything in it was pushed here. This file records where the work stood.

## What it is

CogWorks Week 1: audio identification benchmark: Scores a Week 1 capstone submission on identifying songs from short clips.
A submission enrolls a catalog, then answers queries cut from those songs
after perturbation (short clips, added noise, pitch shift), plus queries cut
from songs that were never enrolled.



## State

- Checkout branch: `fix/audio-diagnostic-causes` at `89f8d948cd`
- Snapshot branch: `archive/progress-20261118` (HEAD plus the uncommitted work listed below)
- Last commits:
- 89f8d94 2026-09-15 fix(diagnostics): stop excluding the tally from a missing song
- 0357a67 2026-09-15 fix(diagnostics): stop reading internals off a candidate list
- a01bba0 2026-09-15 docs: bound the sample-rate evidence to the case it came from
- 73e066e 2026-09-15 fix(diagnostics): name the checks, not a cause this cannot see
- b156644 2026-09-10 feat: say what kind of number each Audio metric is
- b590638 2026-09-04 feat: classify each candidate once, then pair only what can store a song
- a39e56b 2026-09-03 feat: bind the repositories whose database is an argument, an object's table, or several readers deep
- 8d4cd80 2026-09-02 fix: describe the headline score as what the scorer computes

The working tree was clean; nothing uncommitted was captured.

## What's left

Nothing recorded beyond the commits above.

## How to resume

```sh
git clone https://github.com/SamGu-NRX/cogworks-week1-audio-benchmark.git "$HOME/Programming Projects/CogPortal-audio-diagnostics"
git -C "$HOME/Programming Projects/CogPortal-audio-diagnostics" checkout archive/progress-20261118
```

## Thread

No T3 thread was found for this checkout.
