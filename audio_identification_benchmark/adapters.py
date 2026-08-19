"""Bounded resolution of a submission onto the identify protocol.

Only exact documented names are mapped, and only when the callable's arity
can actually accept the call. Everything else raises with a report naming
what was found. That asymmetry is on purpose: requiring a ten-line adapter
from a student is fine, and silently binding ``identify_clip(self, path,
db, top_n)`` to a two-argument call is not. A mis-bound method produces a
number, and a number is indistinguishable from a real result.

Three kinds of target resolve:

- an object with the methods on it (the common case),
- a module, when the repository is functions at module level with no class,
- a dict or list of modules, when the two halves live in different files
  (``database.add`` and ``matching.identify``), which one audited repo does.

Every alias applied and every candidate refused is recorded in ``mappings``
so the run page can show what the platform did rather than only what it
scored.
"""

from __future__ import annotations

import difflib
import inspect
import sys
from types import ModuleType
from typing import Any, Callable, List, Optional, Sequence, Tuple

from .contracts import AdapterContractError

#: Exact names, in preference order. Sourced from the contract itself and
#: from names that appear in audited student repositories.
ENROLL_ALIASES = (
    "enroll",
    "add_song",
    "add_song_to_database",
    "add_to_database",
    "add_fingerprints",
    "register_song",
)
IDENTIFY_ALIASES = (
    "identify",
    "identify_song",
    "identify_clip",
    "recognize",
    "recognize_song",
    "match_song",
    "query_database",
)
FINGERPRINT_ALIASES = ("fingerprint", "fingerprints", "make_fingerprints", "get_fingerprints")
FINALIZE_ALIASES = ("finalize_database", "finalize", "save_database", "commit_database")

#: Parameter names that make a two-audio-argument binding plausible. Used
#: only to demote a candidate, never to promote one.
_AUDIO_HINTS = ("sample", "samples", "audio", "signal", "clip", "data", "y", "array", "arr")
_RATE_HINTS = ("rate", "sr", "fs", "sampling", "samplerate", "sample_rate")

_PROTOCOL_DOC = (
    "Add a submission.py at your repository root exposing enroll(song_id, samples, "
    "sample_rate) and identify(samples, sample_rate). identify returns ranked "
    "(song_id, score) pairs, best first, or [] for no match."
)


class AdaptedIdentifier:
    """The submission behind a uniform surface, with a mapping log."""

    def __init__(
        self,
        target: Any,
        enroll_fn: Callable[..., Any],
        identify_fn: Callable[..., Any],
        fingerprint_fn: Optional[Callable[..., Any]],
        finalize_fn: Optional[Callable[..., Any]],
        mappings: List[str],
    ) -> None:
        self._target = target
        self._enroll_fn = enroll_fn
        self._identify_fn = identify_fn
        self._fingerprint_fn = fingerprint_fn
        self._finalize_fn = finalize_fn
        self.mappings = mappings
        # Instructor-supplied adapters (benchmarks/adapters/) declare what
        # wiring we wrote versus what the team wrote. Carry it through the
        # wrapper so the run result can say so: a public leaderboard must
        # never credit a team for code they did not write.
        #
        # Look on the defining MODULE as well as the object, because these
        # adapters write PROVENANCE at module scope while the factory returns
        # an instance. Checking only the instance silently found nothing.
        self.provenance = getattr(target, "PROVENANCE", None)
        if self.provenance is None:
            module = sys.modules.get(getattr(type(target), "__module__", ""))
            self.provenance = getattr(module, "PROVENANCE", None)

    @property
    def has_fingerprint(self) -> bool:
        return self._fingerprint_fn is not None

    @property
    def has_finalize(self) -> bool:
        return self._finalize_fn is not None

    def enroll(self, song_id: str, samples: Any, sample_rate: int) -> None:
        self._enroll_fn(song_id, samples, sample_rate)

    def identify(self, samples: Any, sample_rate: int) -> Any:
        return self._identify_fn(samples, sample_rate)

    def fingerprint(self, samples: Any, sample_rate: int) -> Any:
        if self._fingerprint_fn is None:
            return None
        return self._fingerprint_fn(samples, sample_rate)

    def finalize_database(self) -> None:
        if self._finalize_fn is not None:
            self._finalize_fn()


def _arity_ok(candidate: Callable[..., Any], required: int) -> Tuple[bool, str]:
    """Whether ``candidate`` can be called with exactly ``required`` positionals.

    A callable that demands more arguments than the contract passes is
    refused rather than called: the failure would otherwise arrive as a
    TypeError from inside student code, attributed to their pipeline instead
    of to our binding. A builtin or C function with no readable signature is
    allowed through, since refusing it would be guessing in the other
    direction.
    """

    try:
        signature = inspect.signature(candidate)
    except (TypeError, ValueError):
        return True, "signature unavailable"
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD, parameter.VAR_POSITIONAL)
    ]
    if any(parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters):
        return True, "accepts *args"
    mandatory = [
        parameter
        for parameter in parameters
        if parameter.default is inspect.Parameter.empty
    ]
    if len(mandatory) > required:
        names = ", ".join(parameter.name for parameter in mandatory)
        return False, "requires {} arguments ({})".format(len(mandatory), names)
    if len(parameters) < required:
        return False, "accepts only {} positional arguments".format(len(parameters))
    return True, "ok"


def _type_plausible(candidate: Callable[..., Any], required: int) -> Tuple[bool, str]:
    """Whether the parameter names look like (audio, rate), roughly.

    Only ever used to refuse. A method whose second parameter is named
    ``database`` or ``top_n`` is not going to do the right thing with an
    integer sample rate, and binding it anyway is how a submission gets a
    score that describes our mistake.
    """

    try:
        signature = inspect.signature(candidate)
    except (TypeError, ValueError):
        return True, "signature unavailable"
    names = [
        parameter.name.lower()
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ][:required]
    if len(names) < required:
        return True, "too few names to judge"
    rate_name = names[-1]
    if any(hint in rate_name for hint in _RATE_HINTS):
        return True, "ok"
    if any(hint in rate_name for hint in _AUDIO_HINTS):
        # Two audio-ish parameters is a comparison function, not identify.
        return False, "second parameter {!r} does not look like a sample rate".format(rate_name)
    return False, "second parameter {!r} does not look like a sample rate".format(rate_name)


def _owners(target: Any) -> List[Tuple[str, Any]]:
    """Every ``(label, object)`` a documented name may be resolved from."""

    if isinstance(target, dict):
        return [(str(key), value) for key, value in target.items()]
    if isinstance(target, (list, tuple)):
        return [
            (getattr(item, "__name__", "item{}".format(index)), item)
            for index, item in enumerate(target)
        ]
    label = getattr(target, "__name__", None) if isinstance(target, ModuleType) else None
    return [(label or type(target).__name__, target)]


def _public_callables(target: Any) -> List[str]:
    names: List[str] = []
    for label, owner in _owners(target):
        for name in dir(owner):
            if name.startswith("_"):
                continue
            try:
                if callable(getattr(owner, name)):
                    names.append(name if len(_owners(target)) == 1 else "{}.{}".format(label, name))
            except Exception:  # noqa: BLE001 - properties may raise; skip them
                continue
    return sorted(set(names))


def _resolve(
    target: Any,
    aliases: Sequence[str],
    role: str,
    arity: int,
    mappings: List[str],
    refusals: List[str],
    check_types: bool = False,
) -> Optional[Callable[..., Any]]:
    """First alias that exists and can accept the call, or ``None``."""

    for label, owner in _owners(target):
        for alias in aliases:
            candidate = getattr(owner, alias, None)
            if not callable(candidate):
                continue
            ok, reason = _arity_ok(candidate, arity)
            if not ok:
                refusals.append(
                    "{} looks like {} but {}; not mapped".format(alias, role, reason)
                )
                continue
            if check_types:
                ok, reason = _type_plausible(candidate, arity)
                if not ok:
                    refusals.append(
                        "{} looks like {} but {}; not mapped".format(alias, role, reason)
                    )
                    continue
            if alias != aliases[0] or len(_owners(target)) > 1:
                mappings.append("mapped {}.{} -> {}".format(label, alias, role))
            return candidate
    return None


def adapt_identifier(target: Any) -> AdaptedIdentifier:
    """Wrap a submission object, module, or module collection; or raise."""

    if isinstance(target, AdaptedIdentifier):
        return target
    mappings: List[str] = []
    refusals: List[str] = []

    enroll_fn = _resolve(target, ENROLL_ALIASES, "enroll", 3, mappings, refusals)
    identify_fn = _resolve(
        target, IDENTIFY_ALIASES, "identify", 2, mappings, refusals, check_types=True
    )
    fingerprint_fn = _resolve(
        target, FINGERPRINT_ALIASES, "fingerprint", 2, mappings, refusals, check_types=True
    )
    finalize_fn = _resolve(target, FINALIZE_ALIASES, "finalize_database", 0, mappings, refusals)

    if enroll_fn is None or identify_fn is None:
        missing = []
        if enroll_fn is None:
            missing.append("enroll(song_id, samples, sample_rate)")
        if identify_fn is None:
            missing.append("identify(samples, sample_rate)")
        available = _public_callables(target)
        report = [
            "received {} with callables: {}".format(
                _describe(target), ", ".join(available) or "none"
            ),
            "missing: {}".format(", ".join(missing)),
        ]
        report.extend(refusals)
        close = _closest(available, missing)
        if close:
            report.append(
                "closest names found: {} (not mapped automatically; only exact "
                "documented names with a plausible signature are).".format(", ".join(close))
            )
        report.append(_PROTOCOL_DOC)
        raise AdapterContractError(
            "The submission does not expose the identify protocol.", report
        )

    mappings.extend(refusals)
    return AdaptedIdentifier(
        target, enroll_fn, identify_fn, fingerprint_fn, finalize_fn, mappings
    )


def _describe(target: Any) -> str:
    if isinstance(target, dict):
        return "a dict of {} modules".format(len(target))
    if isinstance(target, (list, tuple)):
        return "a {} of {} objects".format(type(target).__name__, len(target))
    if isinstance(target, ModuleType):
        return "module {}".format(getattr(target, "__name__", "?"))
    return type(target).__name__


def _closest(available: Sequence[str], missing: Sequence[str]) -> List[str]:
    bare = [name.split(".")[-1] for name in available]
    found: List[str] = []
    for role in missing:
        stem = role.split("(")[0]
        for match in difflib.get_close_matches(stem, bare, n=2, cutoff=0.35):
            if match not in found:
                found.append(match)
    return found[:4]


def instantiate(factory: Any, resources: Any) -> Any:
    """Call the submission factory once, or use the object it already is."""

    if not callable(factory):
        return factory
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(resources)
    try:
        signature.bind(resources)
    except TypeError:
        try:
            signature.bind()
        except TypeError as error:
            raise AdapterContractError(
                "The submission factory must accept the benchmark resources object as "
                "its only required argument.",
                ["factory signature: {}".format(signature), _PROTOCOL_DOC],
            ) from error
        return factory()
    return factory(resources)


def public_callables(target: Any) -> List[str]:
    """The names a mapping report would list. Exposed for staff tooling."""

    return _public_callables(target)
