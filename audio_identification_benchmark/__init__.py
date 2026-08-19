"""CogWorks Week 1 audio benchmark: identify a song from a short clip.

A student submission enrolls a catalog of songs and then answers queries cut
from those songs after perturbation (shorter clips, added noise, pitch
shift), plus queries cut from songs that were never enrolled. The controller
(this package, or the portal runner re-using it) computes every metric from
the ranked candidate lists the submission returns; nothing in the scoring
path runs a reference fingerprinter, so a broken or inert submission lands
at chance rather than at an accidental constant.

The corpus is synthesized from a seeded generator and verified by sha256
before any student code runs, so a laptop and the hosted runner score the
same audio.
"""

__version__ = "0.1.0"
