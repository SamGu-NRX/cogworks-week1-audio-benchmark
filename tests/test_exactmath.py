"""The deterministic transcendentals: accurate enough, and exactly reproducible.

Two things are worth testing here and they pull in opposite directions.

Accuracy is bounded against numpy, because these replace libm inside the
corpus renderer and a wrong sine would render music nobody meant. The bounds
are loose on purpose -- about an ulp -- since being *exactly* numpy is not the
goal and would be impossible anyway; that is the whole reason this module
exists.

Reproducibility cannot be tested on one machine, because the failure it guards
is cross-platform. What can be tested is the property that makes it hold: every
operation used is one IEEE-754 specifies exactly. So the tests below pin the
observable consequences -- exact results on exact inputs, and identical output
for the same input regardless of array shape, memory layout, or batch size,
which is what SIMD-kernel differences perturb in practice.

The end-to-end guard is ``test_synth.py``'s sha256 pins plus the manifest
checks; this file covers the layer beneath them.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio_identification_benchmark import exactmath


def _max_rel(actual, expected):
    expected = np.asarray(expected, dtype=np.float64)
    scale = np.where(np.abs(expected) > 1e-300, np.abs(expected), 1.0)
    return float(np.max(np.abs(np.asarray(actual) - expected) / scale))


class TestAccuracy:
    """Within an ulp or so of numpy, over the ranges the renderer uses."""

    def test_sin_matches_numpy_over_renderer_phase_range(self):
        # A 45 s song at 44.1 kHz with partials to ~5 kHz reaches phases of
        # order 1e6 radians; test an order beyond that.
        for limit in (10.0, 1.0e3, 1.0e6):
            x = np.linspace(-limit, limit, 100_001)
            assert np.max(np.abs(exactmath.sin(x) - np.sin(x))) < 1e-9

    def test_cos_matches_numpy(self):
        x = np.linspace(-1.0e3, 1.0e3, 100_001)
        assert np.max(np.abs(exactmath.cos(x) - np.cos(x))) < 1e-9

    def test_exp_matches_numpy_over_envelope_range(self):
        # Envelopes are exp(-t/tau) with tau >= 0.012 over t <= 0.18, so the
        # argument stays small; the range here is far wider.
        x = np.linspace(-30.0, 5.0, 100_001)
        assert _max_rel(exactmath.exp(x), np.exp(x)) < 1e-14

    def test_exp2_matches_numpy(self):
        x = np.linspace(-20.0, 10.0, 100_001)
        assert _max_rel(exactmath.exp2(x), np.exp2(x)) < 1e-14

    def test_log_matches_numpy_across_the_exponent_range(self):
        x = np.logspace(-30.0, 30.0, 100_001)
        assert _max_rel(exactmath.log(x), np.log(x)) < 1e-14

    def test_power_matches_numpy_for_harmonic_rolloff(self):
        # gains = harmonic_index ** -rolloff, the only use of `power`.
        harmonics = np.arange(1, 32, dtype=np.float64)
        for rolloff in (0.9, 1.5, 2.1):
            assert _max_rel(
                exactmath.power(harmonics, -rolloff), np.power(harmonics, -rolloff)
            ) < 1e-14

    def test_midi_to_hz_error_is_far_below_a_spectrogram_bin(self):
        """The real tolerance: a bin is 10.8 Hz at NFFT=4096, Fs=44100."""

        midi = np.arange(0, 128, dtype=np.float64)
        ours = 440.0 * exactmath.power2((midi - 69.0) / 12.0)
        theirs = 440.0 * np.power(2.0, (midi - 69.0) / 12.0)
        assert np.max(np.abs(ours - theirs)) < 1e-9  # bin width is 10.766 Hz


class TestExactOnExactInputs:
    """Where the true answer is representable, return it exactly."""

    @pytest.mark.parametrize("value", [0.0, 1.0, 3.0, 10.0, -1.0, -5.0])
    def test_exp2_is_exact_on_integers(self, value):
        assert float(exactmath.exp2(np.float64(value))) == 2.0 ** value

    def test_exp_of_zero_is_one(self):
        assert float(exactmath.exp(np.float64(0.0))) == 1.0

    def test_log_of_one_is_zero(self):
        assert float(exactmath.log(np.float64(1.0))) == 0.0

    @pytest.mark.parametrize("value", [1.0, 2.0, 4.0, 0.5, 1024.0])
    def test_log_of_powers_of_two_is_exact(self, value):
        assert float(exactmath.log2(np.float64(value))) == np.log2(value)

    def test_concert_a_is_exactly_440(self):
        """MIDI 69 is A4. An off-by-an-ulp root note would be a bad smell."""

        assert float(440.0 * exactmath.power2(np.float64(0.0))) == 440.0

    def test_sin_of_zero_is_zero(self):
        assert float(exactmath.sin(np.float64(0.0))) == 0.0


class TestShapeIndependence:
    """The property that cross-platform reproducibility actually rests on.

    A libm picks a SIMD kernel by array length and alignment, which is why
    ``np.sin`` can differ between machines and, on some builds, between array
    shapes on one machine. Everything here is elementwise multiply and add, so
    the result must not depend on how the input is batched or laid out. If this
    ever fails, the module has grown a call that dispatches on shape.
    """

    def _reference(self, function, values):
        return np.array([float(function(np.float64(v))) for v in values])

    @pytest.mark.parametrize(
        "function, values",
        [
            (exactmath.sin, np.linspace(-100.0, 100.0, 97)),
            (exactmath.cos, np.linspace(-100.0, 100.0, 97)),
            (exactmath.exp, np.linspace(-10.0, 3.0, 97)),
            (exactmath.exp2, np.linspace(-10.0, 3.0, 97)),
            (exactmath.log, np.linspace(0.01, 100.0, 97)),
        ],
    )
    def test_elementwise_equals_scalar_bit_for_bit(self, function, values):
        batched = np.asarray(function(values), dtype=np.float64)
        scalar = self._reference(function, values)
        assert batched.tobytes() == scalar.tobytes()

    def test_non_contiguous_input_gives_identical_bits(self):
        """Same values, different strides. The view must be interleaved from a
        wider buffer rather than recomputed, or the two inputs differ and the
        test proves nothing about layout."""

        contiguous = np.linspace(-50.0, 50.0, 200)
        wider = np.empty(400, dtype=np.float64)
        wider[::2] = contiguous
        wider[1::2] = 0.0
        strided = wider[::2]
        assert not strided.flags["C_CONTIGUOUS"]
        assert np.asarray(exactmath.sin(contiguous)).tobytes() == (
            np.ascontiguousarray(exactmath.sin(strided)).tobytes()
        )

    def test_two_dimensional_input_matches_its_flat_form(self):
        flat = np.linspace(-20.0, 20.0, 120)
        assert np.asarray(exactmath.sin(flat.reshape(10, 12))).ravel().tobytes() == (
            np.asarray(exactmath.sin(flat)).tobytes()
        )


class TestArgumentReduction:
    """The reduction is where a naive implementation loses its low bits."""

    def test_large_arguments_keep_full_precision(self):
        """sin(x + 2pi*k) == sin(x) is the reduction's whole job."""

        base = np.array([0.3, 1.1, 2.7, -0.9])
        for k in (1, 100, 10_000):
            shifted = base + 2.0 * np.pi * k
            assert np.max(np.abs(exactmath.sin(shifted) - exactmath.sin(base))) < 1e-9

    def test_reduction_survives_the_renderer_worst_case(self):
        """The highest phase a rendered song reaches, checked against numpy."""

        # 45 s at the highest audible partial (~21.6 kHz): 2*pi*f*t.
        phase = 2.0 * np.pi * 21_600.0 * np.linspace(0.0, 45.0, 50_001)
        assert np.max(np.abs(exactmath.sin(phase) - np.sin(phase))) < 1e-7
