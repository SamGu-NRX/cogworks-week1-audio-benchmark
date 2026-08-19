"""Transcendental functions that produce the same bits on every machine.

IEEE-754 mandates exact results for ``+ - * /`` and ``sqrt`` and says nothing
about ``sin``, ``exp``, or ``pow``. Every libm is free to differ in the last
bits, and they do. Rendering one song of the Week 1 corpus under numpy 1.24 on
Linux/x86 and under numpy 2.4 on macOS/arm64 produces different sha256 values,
because ``np.sin`` of the same float64 array differs between them. Measured
with ``apps/runner-modal/tools/diagnose_corpus.py``:

    rng_normal    identical        (the RNG was never the problem)
    rng_integers  identical
    cumsum        identical        (add and multiply are exact)
    sin_f64       DIFFERS
    exp_f64       DIFFERS
    pow_f64       DIFFERS
    render_song   DIFFERS

That broke the corpus contract outright. The manifest pins a sha256 per song
and the sandbox re-renders and verifies it before student code runs, so a
platform difference in libm surfaced as a corrupted-corpus failure and the
hosted path could not score anyone. ``synth.py`` had claimed float64 was
sufficient for determinism; float64 fixes the *rounding* of each operation but
not *which* operation a given libm performs, and only the second one matters
here.

Everything below is built from multiply, add, subtract, ``rint``, and
``ldexp``, all of which IEEE-754 pins exactly, so results are bit-identical
anywhere numpy runs.

These are not claimed to be more accurate than libm. They are about an ulp
less accurate, and that is the right trade: the corpus is *defined* as what
this renderer emits, so there is no external truth a better sine would be
closer to. What the corpus must be is identical everywhere, which libm is not.

Accuracy is measured in ``tests/test_exactmath.py`` rather than asserted here.
Roughly: relative error near 1e-16 for all three over the ranges the renderer
uses -- about nine orders of magnitude below the float32 cast the corpus ends
in, and further still below the log-magnitude spectrogram a fingerprinter
actually sees.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sin",
    "cos",
    "exp",
    "exp2",
    "log",
    "log2",
    "power",
    "power2",
    "box_average",
    "interp",
    "uniform",
]

# --------------------------------------------------------------------------
# Argument reduction constants
# --------------------------------------------------------------------------

#: 2*pi split into three float64 parts. The high part has its low 27 mantissa
#: bits cleared, so `k * _TWO_PI_HI` is exact (no rounding) for every |k| this
#: renderer reaches. Subtracting the parts in sequence is then a Cody-Waite
#: reduction: cancellation happens against the exact term first, so the low
#: bits of x survive into the reduced argument instead of being lost.
_TWO_PI_HI = 6.283185243606567
_TWO_PI_LO = 6.357301909411278e-08
_TWO_PI_LO2 = 2.54732686540438e-24

_INV_TWO_PI = 0.15915494309189535

#: pi/2, same three-part split, for cos.
_HALF_PI_HI = 1.5707963109016418
_HALF_PI_LO = 1.5893254773199803e-08
_HALF_PI_LO2 = 6.368317163510951e-25

#: Taylor coefficients of (sin(x) - x) / x^3 = sum (-1)^(n+1) x^2n / (2n+3)!,
#: highest power first, through x^28. On the reduced range |r| <= pi the tail
#: after this many terms is below 1e-30 relative, so the series is exact to
#: float64 and the coefficients are just reciprocal factorials -- no minimax
#: fit to audit, and each one is checkable by hand.
_SIN_COEFFS = (
    -9.183689863795546e-29,
    6.446950284384474e-26,
    -3.868170170630684e-23,
    1.9572941063391263e-20,
    -8.22063524662433e-18,
    2.8114572543455206e-15,
    -7.647163731819816e-13,
    1.6059043836821613e-10,
    -2.505210838544172e-08,
    2.7557319223985893e-06,
    -0.0001984126984126984,
    0.008333333333333333,
    -0.16666666666666666,
)

#: Taylor coefficients of (exp(r) - 1) / r = sum r^(n-1) / n!, highest power
#: first, n = 16 down to 1. Used on |r| <= ln(2)/2 = 0.347, where the tail
#: after 16 terms is below 1e-22 relative.
_EXP_COEFFS = (
    4.779477332387385e-14,
    7.647163731819816e-13,
    1.1470745597729725e-11,
    1.6059043836821613e-10,
    2.08767569878681e-09,
    2.505210838544172e-08,
    2.755731922398589e-07,
    2.7557319223985893e-06,
    2.48015873015873e-05,
    0.0001984126984126984,
    0.001388888888888889,
    0.008333333333333333,
    0.041666666666666664,
    0.16666666666666666,
    0.5,
    1.0,
)

#: ln(2), split the same way so `whole * _LN2_HI` is exact.
_LN2_HI = 0.6931471675634384
_LN2_LO = 1.2996506893889889e-08

_LOG2_E = 1.4426950408889634


def _horner(coefficients, x):
    """Polynomial evaluation using only multiply and add."""

    result = np.full(np.shape(x), coefficients[0], dtype=np.float64)
    for coefficient in coefficients[1:]:
        result = result * x + coefficient
    return result


def sin(x):
    """``np.sin``, bit-identical on every platform.

    Reduce ``x`` to ``r`` in about [-pi, pi] by subtracting ``k * 2pi`` in
    three exact steps, then evaluate the Taylor series in ``r``. Both halves
    use only exactly-specified operations.
    """

    x = np.asarray(x, dtype=np.float64)
    k = np.rint(x * _INV_TWO_PI)
    reduced = x - k * _TWO_PI_HI
    reduced = reduced - k * _TWO_PI_LO
    reduced = reduced - k * _TWO_PI_LO2
    squared = reduced * reduced
    # r + r^3 * P(r^2), written so the small correction is added to r last and
    # keeps its low bits rather than being absorbed.
    return reduced + (reduced * squared) * _horner(_SIN_COEFFS, squared)


def cos(x):
    """``np.cos``, via ``cos(x) = sin(x + pi/2)`` with the exact pi/2 split.

    Adding a rounded pi/2 to x first would throw away the low bits of the sum
    before reduction ever sees them, so the shift is folded into the reduction
    instead.
    """

    x = np.asarray(x, dtype=np.float64)
    k = np.rint(x * _INV_TWO_PI)
    reduced = x - k * _TWO_PI_HI
    reduced = reduced - k * _TWO_PI_LO
    reduced = reduced - k * _TWO_PI_LO2
    shifted = ((reduced + _HALF_PI_HI) + _HALF_PI_LO) + _HALF_PI_LO2
    # The shift can push past pi; one more reduction step brings it back.
    j = np.rint(shifted * _INV_TWO_PI)
    shifted = shifted - j * _TWO_PI_HI
    shifted = shifted - j * _TWO_PI_LO
    shifted = shifted - j * _TWO_PI_LO2
    squared = shifted * shifted
    return shifted + (shifted * squared) * _horner(_SIN_COEFFS, squared)


def exp(x):
    """``np.exp``, bit-identical on every platform.

    Write ``x = n*ln2 + r`` with ``|r| <= ln2/2``. ``exp(r)`` comes from the
    Taylor series and the ``2**n`` factor from ``ldexp``, which only writes the
    exponent field and is therefore exact.
    """

    x = np.asarray(x, dtype=np.float64)
    whole = np.rint(x * _LOG2_E)
    reduced = x - whole * _LN2_HI
    reduced = reduced - whole * _LN2_LO
    mantissa = 1.0 + reduced * _horner(_EXP_COEFFS, reduced)
    return np.ldexp(mantissa, whole.astype(np.int64))


def exp2(x):
    """``2.0 ** x``. Exact for integer ``x`` because the fraction is then 0."""

    x = np.asarray(x, dtype=np.float64)
    whole = np.rint(x)
    fraction = x - whole
    reduced = fraction * _LN2_HI + fraction * _LN2_LO
    mantissa = 1.0 + reduced * _horner(_EXP_COEFFS, reduced)
    return np.ldexp(mantissa, whole.astype(np.int64))


def power2(x):
    """``np.power(2.0, x)``. Named apart from ``exp2`` to read at call sites."""

    return exp2(x)


#: Taylor coefficients of atanh(s)/s = sum s^2n/(2n+1), highest power first,
#: through s^28. On |s| <= (sqrt2-1)/(sqrt2+1) = 0.1716 the tail is below
#: 1e-24 relative.
_ATANH_COEFFS = tuple(1.0 / (2 * n + 1) for n in range(14, -1, -1))


def log(x):
    """``np.log``, bit-identical on every platform.

    ``frexp`` splits ``x`` into mantissa and exponent exactly, then
    ``log(m) = 2*atanh((m-1)/(m+1))`` converges fast once ``m`` is nudged into
    ``[sqrt(1/2), sqrt(2))``, where ``|s| <= 0.172``.
    """

    x = np.asarray(x, dtype=np.float64)
    mantissa, exponent = np.frexp(x)  # x == mantissa * 2**exponent, m in [0.5,1)
    # Move mantissa into [sqrt(1/2), sqrt(2)) so the series argument is small.
    small = mantissa < 0.7071067811865476
    mantissa = np.where(small, mantissa * 2.0, mantissa)
    exponent = np.where(small, exponent - 1, exponent).astype(np.float64)
    s = (mantissa - 1.0) / (mantissa + 1.0)
    squared = s * s
    log_mantissa = 2.0 * (s + s * squared * _horner(_ATANH_COEFFS[:-1], squared))
    return (exponent * _LN2_HI + exponent * _LN2_LO) + log_mantissa


def log2(x):
    """``np.log2``, bit-identical on every platform."""

    return log(x) * _LOG2_E


def power(base, exponent):
    """``np.power(base, exponent)`` for strictly positive ``base``.

    ``base**e == exp(e * log(base))``. Undefined for non-positive ``base``;
    the renderer only ever raises positive harmonic indices.
    """

    return exp(np.asarray(exponent, dtype=np.float64) * log(base))


# --------------------------------------------------------------------------
# Array operations that are not transcendental but are still not reproducible
# --------------------------------------------------------------------------
#
# `np.convolve` and `np.interp` gave different bits on identical inputs
# between the two builds, measured the same way as the libm functions above:
#
#     convolve_48   DIFFERS      convolve_9   DIFFERS
#     interp        DIFFERS
#     sum/mean/dot  identical    (plain reductions were never the problem)
#
# Neither is a rounding-mode question. `np.convolve` runs a correlate kernel
# whose blocking and accumulation order depend on the numpy build, and
# `np.interp` changed its internal slope formulation between 1.24 and 2.x.
# Both are replaced here with fixed-order arithmetic.


def box_average(signal, width):
    """Length-preserving box filter, with a fixed summation order.

    Equivalent to ``np.convolve(signal, ones(width)/width, mode="same")`` up to
    the last bits, computed from a prefix sum so each output is one subtraction
    of two partial sums rather than a dot product whose blocking the numpy
    build chooses. Reproducible because ``cumsum`` is strictly sequential.
    """

    signal = np.asarray(signal, dtype=np.float64)
    width = int(width)
    if width <= 1 or signal.size == 0:
        return signal.copy()
    if width > signal.size:
        # np.convolve(mode="same") returns max(len(signal), width) elements, so
        # a kernel longer than the signal would change the array's length. The
        # renderer never does this (its longest kernel is 48 against a 0.18 s
        # hit, ~7900 samples), and silently returning a different length would
        # be worse than saying so.
        raise ValueError(
            "box_average needs width <= len(signal); got width={} for {} samples.".format(
                width, signal.size
            )
        )

    count = signal.size
    # `mode="same"` centres the window as `full[(width - 1) // 2 :][:count]`,
    # which for an even width puts the extra sample on the LEFT of the output
    # index, not the right. Getting this backwards shifts every even-width
    # result by one sample -- inaudible, but a different corpus.
    after = (width - 1) // 2
    before = width - 1 - after
    prefix = np.concatenate(([0.0], np.cumsum(signal)))
    index = np.arange(count)
    high = np.minimum(index + after + 1, count)
    low = np.maximum(index - before, 0)
    # Divide by the full width, not by the number of samples in range, because
    # convolve zero-pads rather than shrinking the window at the edges.
    return (prefix[high] - prefix[low]) / float(width)


def uniform(rng, low, high, size=None):
    """``rng.uniform``, with a fixed formula and a fixed number of RNG draws.

    ``Generator.uniform`` is not stable across numpy versions: seeded with the
    same value, numpy 1.24 and 2.4 return different doubles, while
    ``standard_normal``, ``integers``, and ``random`` from the same generator
    return identical bits. So the bit stream is fine and the scaling is not.

    ``random()`` is the stable primitive, and ``low + (high - low) * u`` is one
    specific ordering of the scaling -- the point is not that it is the best
    ordering but that it is pinned here rather than chosen by whichever numpy
    the container happens to carry. One ``random()`` draw per value, so the
    generator advances exactly as ``uniform`` did and every later draw in the
    same song is unchanged.
    """

    draw = rng.random() if size is None else rng.random(size)
    return low + (high - low) * draw


def interp(positions, x_points, y_points):
    """``np.interp`` for a sorted, unit-spaced ``x_points``, written out.

    The renderer only ever resamples against ``arange(n)``, so the bracketing
    index is ``floor(position)`` and the local fraction is the remainder. That
    makes this a two-term blend with no search, and the blend is written as
    ``y0 + f*(y1 - y0)`` -- one specific ordering, rather than whichever of the
    algebraically equal forms a given numpy release chose.
    """

    positions = np.asarray(positions, dtype=np.float64)
    y_points = np.asarray(y_points, dtype=np.float64)
    count = y_points.size
    if count == 0:
        raise ValueError("interp needs at least one sample point.")
    if count == 1:
        return np.full(positions.shape, y_points[0], dtype=np.float64)

    expected = np.arange(count, dtype=np.float64)
    if x_points is not None and not np.array_equal(
        np.asarray(x_points, dtype=np.float64), expected
    ):
        raise ValueError(
            "exactmath.interp only handles x_points == arange(len(y_points)); "
            "got a different grid."
        )

    clamped = np.clip(positions, 0.0, float(count - 1))
    lower = np.floor(clamped).astype(np.int64)
    lower = np.minimum(lower, count - 2)
    fraction = clamped - lower.astype(np.float64)
    low = y_points[lower]
    high = y_points[lower + 1]
    return low + fraction * (high - low)
