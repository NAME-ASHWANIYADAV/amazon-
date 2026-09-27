import numpy as np
import pytest

from src.run_pipeline import band_shift


def test_band_shift_moves_mid_band_and_keeps_extremes():
    knots = [[0.4, -0.7], [0.85, -2.3], [0.97, -2.5], [0.995, 0.0]]
    p = np.array([0.01, 0.4, 0.85, 0.97, 0.995, 0.9999], dtype=np.float32)
    q = band_shift(p, knots)
    assert q[0] < 0.01 and q[0] > 0.004          # constant shift below the first knot
    assert abs(q[1] - 1 / (1 + np.exp(-(np.log(0.4 / 0.6) - 0.7)))) < 1e-5
    assert q[2] < 0.5 and q[3] < 0.8              # mid band pushed down
    assert q[4] == pytest.approx(0.995, abs=1e-4) and q[5] == pytest.approx(0.9999, abs=1e-4)  # top band untouched


def test_band_shift_is_monotone():
    knots = [[0.4, -0.7], [0.6, -1.46], [0.75, -1.85], [0.85, -2.31], [0.925, -2.53], [0.97, -2.55], [0.995, 0.0]]
    p = np.linspace(0.001, 0.9999, 5000).astype(np.float32)
    q = band_shift(p, knots)
    assert np.all(np.diff(q) >= -1e-6)


def test_band_shift_rejects_non_monotone_knots():
    with pytest.raises(ValueError):
        band_shift(np.array([0.5], dtype=np.float32), [[0.5, 0.0], [0.6, -1.0]])
