"""Every accepted return shape normalizes; everything else names its type."""

from __future__ import annotations

import numpy as np
import pytest

from audio_identification_benchmark.checks import (
    CheckFailure,
    coerce_candidates,
    normalize_song_id,
)


def test_ranked_pairs_is_the_documented_shape():
    result = coerce_candidates([("song-01", 42.0), ("song-02", 8.0)])
    assert result.ids == ["song-01", "song-02"]
    assert result.scores == [42.0, 8.0]
    assert result.shape == "ranked_pairs"


def test_ranked_ids_has_no_scores():
    result = coerce_candidates(["song-01", "song-02"])
    assert result.ids == ["song-01", "song-02"]
    assert result.scores is None
    assert result.shape == "ranked_ids"


def test_single_id_becomes_a_one_element_list():
    result = coerce_candidates("song-03")
    assert result.ids == ["song-03"]
    assert result.scores is None
    assert result.shape == "single_id"


def test_none_is_no_match():
    result = coerce_candidates(None)
    assert result.ids == []
    assert result.shape == "single_id"


def test_triple_takes_the_id_from_element_zero():
    result = coerce_candidates([("song-01", "Some Artist", 17.0)])
    assert result.ids == ["song-01"]
    assert result.scores == [17.0]
    assert result.shape == "ranked_triples"


def test_empty_list_is_no_match():
    result = coerce_candidates([])
    assert result.ids == []
    assert result.scores == []


@pytest.mark.parametrize("sentinel", ["Unknown", "unknown", "", "None", "No match", "  "])
def test_sentinels_mean_no_match(sentinel):
    assert normalize_song_id(sentinel) is None
    assert coerce_candidates(sentinel).ids == []


def test_sentinel_inside_a_ranked_list_is_dropped_not_ranked():
    result = coerce_candidates([("Unknown", 3.0), ("song-02", 1.0)])
    assert result.ids == ["song-02"]
    assert result.scores == [1.0]


def test_numpy_scalars_and_arrays_are_accepted():
    result = coerce_candidates(np.array([["song-01", "5.0"], ["song-02", "1.0"]]))
    assert result.ids == ["song-01", "song-02"]
    assert result.scores == [5.0, 1.0]
    assert coerce_candidates(np.str_("song-09")).ids == ["song-09"]


def test_numpy_float_scores_are_accepted():
    result = coerce_candidates([("song-01", np.float32(3.5)), ("song-02", np.int64(2))])
    assert result.scores == [3.5, 2.0]


@pytest.mark.parametrize(
    "value, fragment",
    [
        (17, "int"),
        (3.5, "float"),
        ({"song-01": 5}, "dict"),
        ({"song-01", "song-02"}, "set"),
        (object(), "object"),
    ],
)
def test_refused_shapes_name_the_observed_type(value, fragment):
    with pytest.raises(CheckFailure) as info:
        coerce_candidates(value, "identify")
    message = str(info.value)
    assert fragment in message
    assert "song_id, score" in message


def test_a_four_element_row_is_refused_with_its_length():
    with pytest.raises(CheckFailure) as info:
        coerce_candidates([("song-01", "artist", 1.0, "extra")])
    assert "4-element" in str(info.value)


def test_mixed_shapes_in_one_list_are_refused():
    with pytest.raises(CheckFailure) as info:
        coerce_candidates(["song-01", ("song-02", 1.0)])
    assert "mixed result shapes" in str(info.value)


def test_a_non_numeric_score_is_refused():
    with pytest.raises(CheckFailure) as info:
        coerce_candidates([("song-01", "very confident")])
    assert "must be a number" in str(info.value)


def test_a_non_finite_score_is_refused():
    with pytest.raises(CheckFailure) as info:
        coerce_candidates([("song-01", float("nan"))])
    assert "finite" in str(info.value)


def test_an_integer_song_id_is_refused_by_name():
    with pytest.raises(CheckFailure) as info:
        coerce_candidates([(7, 1.0)])
    assert "int" in str(info.value)


def test_already_normalized_values_pass_through():
    once = coerce_candidates([("song-01", 1.0)])
    assert coerce_candidates(once) is once
