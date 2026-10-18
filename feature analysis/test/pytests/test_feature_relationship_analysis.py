import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import src.feature_relationship_analysis as fra
from src.feature_relationship_analysis import (
    resolve_features,
    pairwise_regression_matrix,
    plot_heatmap,
    plot_significant_heatmap,
)


def test_resolve_features_only_keeps_numeric_existing_columns():
    df = pd.DataFrame({
        "x": [1.0, 2.0],
        "y": ["a", "b"],
    })

    out = resolve_features(df, ["x", "y", "z"])

    assert out == ["x"]


def test_pairwise_regression_matrix_returns_expected_shapes_and_values(monkeypatch):
    monkeypatch.setattr(fra, "MIN_ROWS", 4)

    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6],
        "f2": [2, 3, 5, 7, 11, 13],
        "group_binary": [0, 0, 0, 1, 1, 1],
    })

    coef_mat, t_mat, p_mat, long_df = pairwise_regression_matrix(df, ["f1", "f2"])

    assert coef_mat.shape == (2, 2)
    assert t_mat.shape == (2, 2)
    assert p_mat.shape == (2, 2)

    assert np.isnan(coef_mat.loc["f1", "f1"])
    assert np.isnan(t_mat.loc["f2", "f2"])
    assert np.isnan(p_mat.loc["f1", "f1"])

    assert len(long_df) == 2
    assert set(long_df["feature_x"]) == {"f1", "f2"}
    assert set(long_df["feature_y"]) == {"f1", "f2"}

    assert np.isfinite(coef_mat.loc["f2", "f1"])
    assert np.isfinite(t_mat.loc["f2", "f1"])
    assert np.isfinite(p_mat.loc["f2", "f1"])

    assert np.isfinite(coef_mat.loc["f1", "f2"])
    assert np.isfinite(t_mat.loc["f1", "f2"])
    assert np.isfinite(p_mat.loc["f1", "f2"])


def test_pairwise_regression_matrix_skips_when_group_binary_has_one_level(monkeypatch):
    monkeypatch.setattr(fra, "MIN_ROWS", 4)

    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5],
        "f2": [5, 4, 3, 2, 1],
        "group_binary": [1, 1, 1, 1, 1],
    })

    coef_mat, t_mat, p_mat, long_df = pairwise_regression_matrix(df, ["f1", "f2"])

    assert long_df.empty
    assert np.isnan(coef_mat.loc["f2", "f1"])
    assert np.isnan(t_mat.loc["f2", "f1"])
    assert np.isnan(p_mat.loc["f2", "f1"])


def test_plot_heatmap_writes_png(tmp_path):
    matrix = pd.DataFrame(
        [[np.nan, 1.5], [-2.0, np.nan]],
        index=["f1", "f2"],
        columns=["f1", "f2"],
    )

    out = tmp_path / "heatmap.png"
    plot_heatmap(matrix, "Test heatmap", out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_significant_heatmap_writes_png(tmp_path):
    t_mat = pd.DataFrame(
        [[np.nan, 3.2], [0.8, np.nan]],
        index=["f1", "f2"],
        columns=["f1", "f2"],
    )
    p_mat = pd.DataFrame(
        [[np.nan, 0.001], [0.50, np.nan]],
        index=["f1", "f2"],
        columns=["f1", "f2"],
    )

    out = tmp_path / "sig_heatmap.png"
    plot_significant_heatmap(t_mat, p_mat, p_thresh=0.01, output_path=out)

    assert out.exists()
    assert out.stat().st_size > 0