"""Resolution against objects, modules, and module collections; arity guards."""

from __future__ import annotations

import types

import numpy as np
import pytest

from audio_identification_benchmark.adapters import adapt_identifier, instantiate
from audio_identification_benchmark.contracts import AdapterContractError, Resources

from .fixtures.submissions import ReferenceIdentifier


def _module(name, **members):
    module = types.ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    return module


def test_native_object_passes_through_with_no_mappings():
    adapter = adapt_identifier(ReferenceIdentifier())
    assert adapter.mappings == []
    assert not adapter.has_finalize
    assert not adapter.has_fingerprint


def test_documented_aliases_are_mapped_and_logged():
    class CourseStyle:
        def add_song(self, song_id, samples, sample_rate):
            self.last = song_id

        def recognize(self, samples, sample_rate):
            return [("song-00", 1.0)]

    adapter = adapt_identifier(CourseStyle())
    joined = " ".join(adapter.mappings)
    assert "add_song" in joined and "enroll" in joined
    assert "recognize" in joined and "identify" in joined
    assert adapter.identify(np.zeros(4, dtype=np.float32), 44100) == [("song-00", 1.0)]


def test_module_level_functions_resolve():
    calls = []

    module = _module(
        "student_pkg",
        enroll=lambda song_id, samples, rate: calls.append(song_id),
        identify=lambda samples, sample_rate: [("song-01", 3.0)],
    )
    adapter = adapt_identifier(module)
    adapter.enroll("song-01", np.zeros(4, dtype=np.float32), 44100)
    assert calls == ["song-01"]
    assert adapter.identify(np.zeros(4, dtype=np.float32), 44100)[0][0] == "song-01"


def test_two_modules_supplied_as_a_dict_resolve_separately():
    database = _module("database", add_song=lambda song_id, samples, rate: None)
    matching = _module("matching", identify_song=lambda samples, sample_rate: ["song-02"])
    adapter = adapt_identifier({"database": database, "matching": matching})
    joined = " ".join(adapter.mappings)
    assert "database.add_song" in joined
    assert "matching.identify_song" in joined
    assert adapter.identify(np.zeros(4, dtype=np.float32), 44100) == ["song-02"]


def test_a_list_of_modules_resolves_too():
    database = _module("db", enroll=lambda a, b, c: None)
    matching = _module("match", identify=lambda samples, rate: [])
    adapter = adapt_identifier([database, matching])
    assert adapter.identify(np.zeros(4, dtype=np.float32), 44100) == []


def test_three_arity_identify_clip_is_demoted_not_mis_bound():
    """The measured hazard: identify_clip(clip, database, top_n).

    Binding it to identify(samples, sample_rate) would pass 44100 as the
    database and produce a number that describes our mistake.
    """

    class Repo:
        def enroll(self, song_id, samples, sample_rate):
            pass

        def identify_clip(self, clip, database, top_n):
            return [(database, "artist", float(top_n))]

    with pytest.raises(AdapterContractError) as info:
        adapt_identifier(Repo())
    report = " ".join(info.value.report)
    assert "identify_clip" in report
    assert "requires 3 arguments" in report
    assert "not mapped" in report
    assert "submission.py" in report


def test_a_second_parameter_that_is_not_a_rate_is_demoted():
    class Repo:
        def enroll(self, song_id, samples, sample_rate):
            pass

        def identify(self, clip_samples, database):
            return ["song-00"]

    with pytest.raises(AdapterContractError) as info:
        adapt_identifier(Repo())
    report = " ".join(info.value.report)
    assert "does not look like a sample rate" in report


def test_optional_trailing_arguments_are_fine():
    class Repo:
        def enroll(self, song_id, samples, sample_rate):
            pass

        def identify(self, samples, sample_rate, top_n=5):
            return [("song-00", 1.0)]

    adapter = adapt_identifier(Repo())
    assert adapter.identify(np.zeros(4, dtype=np.float32), 44100)


def test_var_positional_is_accepted():
    class Repo:
        def enroll(self, *args):
            pass

        def identify(self, *args):
            return []

    assert adapt_identifier(Repo()).identify(np.zeros(4, dtype=np.float32), 44100) == []


def test_unmappable_object_gets_an_actionable_report():
    class Mystery:
        def do_the_thing(self, stuff):
            return stuff

    with pytest.raises(AdapterContractError) as info:
        adapt_identifier(Mystery())
    report = " ".join(info.value.report)
    assert "Mystery" in report
    assert "do_the_thing" in report
    assert "enroll(song_id, samples, sample_rate)" in report
    assert "identify(samples, sample_rate)" in report


def test_optional_surfaces_are_detected_when_present():
    class Rich(ReferenceIdentifier):
        def fingerprint(self, samples, sample_rate):
            return [(("a", "b", 1), 0)]

        def finalize_database(self):
            self.finalized = True

    adapter = adapt_identifier(Rich())
    assert adapter.has_fingerprint
    assert adapter.has_finalize
    adapter.finalize_database()
    assert adapter._target.finalized is True


def test_factory_receiving_resources_is_called_with_them():
    seen = {}

    def factory(resources):
        seen["resources"] = resources
        return ReferenceIdentifier()

    resources = Resources()
    assert isinstance(instantiate(factory, resources), ReferenceIdentifier)
    assert seen["resources"] is resources


def test_zero_argument_factory_is_accepted():
    assert isinstance(instantiate(lambda: ReferenceIdentifier(), Resources()), ReferenceIdentifier)


def test_a_factory_needing_two_arguments_is_refused_by_name():
    def factory(resources, extra):
        return None

    with pytest.raises(AdapterContractError) as info:
        instantiate(factory, Resources())
    assert "only required argument" in str(info.value)


def test_an_object_that_is_not_callable_is_used_directly():
    identifier = ReferenceIdentifier()
    assert instantiate(identifier, Resources()) is identifier
