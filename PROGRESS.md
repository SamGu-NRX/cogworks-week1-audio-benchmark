# Progress snapshot: cogworks-week1-audio-care (20261118)

The local checkout `~/Programming Projects/cogworks-week1-audio-care` was removed on 2026-11-18 to free disk space on Sam's Mac, after everything in it was pushed here. This file records where the work stood.

## What it is

CogWorks Week 1: audio identification benchmark: Scores a Week 1 capstone submission on identifying songs from short clips.
A submission enrolls a catalog, then answers queries cut from those songs
after perturbation (short clips, added noise, pitch shift), plus queries cut
from songs that were never enrolled.



## State

- Checkout branch: `care/discovered-chain-reads-the-repository` at `4e516f39ff`
- Snapshot branch: `archive/progress-20261118-cogworks-week1-audio-care` (HEAD plus the uncommitted work listed below)
- Last commits:
- 4e516f3 2026-09-29 test: enumerate predicate candidates inside a live Project
- ad05587 2026-09-18 test: read the scratch repository instead of the resolver's released namespace
- b156644 2026-09-10 feat: say what kind of number each Audio metric is
- b590638 2026-09-04 feat: classify each candidate once, then pair only what can store a song
- a39e56b 2026-09-03 feat: bind the repositories whose database is an argument, an object's table, or several readers deep
- 8d4cd80 2026-09-02 fix: describe the headline score as what the scorer computes
- 5f7ab8e 2026-08-19 feat: turn a resolved repository into the submission the driver runs
- eb262ae 2026-08-19 feat: run a discovered binding, and prove it scores what a human wired

The working tree was clean; nothing uncommitted was captured.

## What's left

Nothing recorded beyond the commits above.

## How to resume

```sh
git clone https://github.com/SamGu-NRX/cogworks-week1-audio-benchmark "$HOME/Programming Projects/cogworks-week1-audio-care"
git -C "$HOME/Programming Projects/cogworks-week1-audio-care" checkout archive/progress-20261118-cogworks-week1-audio-care
```

## Thread

No T3 thread was found for this checkout.
