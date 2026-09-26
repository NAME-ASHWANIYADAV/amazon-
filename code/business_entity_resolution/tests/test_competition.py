import numpy as np
import pytest

from src.features import COMP_COLS, competition_features


def test_competition_features_per_sx_group():
    # rows: (S1 A, sx 1), (S1 B, sx 1), (S1 C, sx 1), (S1 D, sx 2)
    sx = np.array([1, 1, 1, 2])
    cos = np.array([0.90, 0.95, 0.94, 0.50], dtype=np.float32)
    name = np.array([0.80, 0.90, 0.99, 0.50], dtype=np.float32)
    f = competition_features(sx, cos, name)
    col = {c: i for i, c in enumerate(COMP_COLS)}
    assert f[:, col["comp_n_other"]].tolist() == [2, 2, 2, 0]
    assert f[:, col["comp_rank"]].tolist() == [2, 0, 1, 0]
    assert f[:, col["comp_is_best"]].tolist() == [0, 1, 0, 1]
    assert f[:, col["comp_n_close"]].tolist() == [2, 1, 1, 0]
    assert f[:, col["comp_margin"]] == pytest.approx([-0.05, 0.01, -0.01, 1.0], abs=1e-6)
    assert f[:, col["comp_margin_name"]] == pytest.approx([-0.19, -0.09, 0.09, 1.0], abs=1e-6)
