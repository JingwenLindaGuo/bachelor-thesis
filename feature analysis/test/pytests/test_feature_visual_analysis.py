import json
import pandas as pd
import numpy as np

from src.feature_visual_analysis import (
    load_group_json,
    coerce_numeric,
    resolve_plot_features,
    compute_feature_ranges,
    format_tick_values,
)


def test_load_group_json_supports_nested_covariates(tmp_path):
    path = tmp_path / "sample.json"
    payload = [
        {
            "id": "001",
            "text": "abc",
            "covariates": {"group": "depressed", "age": 22},
            "features": {"semantic_coherence": 0.31}
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    df = load_group_json(str(path), fallback_group_label=None)

    assert len(df) == 1
    assert df.loc[0, "group"] == "depressed"
    assert df.loc[0, "age"] == 22
    assert df.loc[0, "semantic_coherence"] == 0.31


def test_load_group_json_uses_fallback_group_label(tmp_path):
    path = tmp_path / "sample.json"
    payload = [{"id": "001", "text": "abc", "features": {"LCC": 12}}]
    path.write_text(json.dumps(payload), encoding="utf-8")

    df = load_group_json(str(path), fallback_group_label="normal")

    assert df.loc[0, "group"] == "normal"


def test_coerce_numeric_converts_numeric_strings_and_invalid_to_nan():
    df = pd.DataFrame({
        "a": ["1", "2", "x"],
        "b": ["3.5", None, "4.5"],
    })

    out = coerce_numeric(df.copy(), ["a", "b"])

    assert out["a"].dtype.kind in "fi"
    assert np.isnan(out.loc[2, "a"])
    assert out.loc[0, "b"] == 3.5


def test_resolve_plot_features_only_keeps_numeric_existing_columns():
    df = pd.DataFrame({
        "x": [1.0, 2.0],
        "y": ["a", "b"],
    })

    out = resolve_plot_features(df, ["x", "y", "z"])

    assert out == ["x"]


def test_compute_feature_ranges_expands_constant_feature():
    df = pd.DataFrame({
        "a": [5.0, 5.0, 5.0],
        "b": [1.0, 3.0, 5.0],
    })

    s_vec, e_vec = compute_feature_ranges(df, ["a", "b"])

    assert s_vec.shape == (2,)
    assert e_vec.shape == (2,)
    assert s_vec[0] < e_vec[0]
    assert s_vec[1] == 1.0
    assert e_vec[1] == 5.0


def test_format_tick_values_formats_by_magnitude():
    values = [123.4, 12.34, 1.234]
    out = format_tick_values(values)
    assert out == ["123", "12.3", "1.23"]