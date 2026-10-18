import json
import numpy as np
import pandas as pd

import src.covariate_sensitivity as cs
from src.covariate_sensitivity import (
    _as_bool,
    load_stage1_normality_map,
    normalise_context_value,
    column_is_numeric_like,
    get_requested_covariate_names,
    prepare_model_covariates,
    build_model_formula,
    validate_feature_data,
    gamma_suitability_report,
    fit_feature_model,
)


def test_as_bool_parses_common_inputs():
    assert _as_bool(True) is True
    assert _as_bool(False) is False
    assert _as_bool(1) is True
    assert _as_bool(0) is False
    assert _as_bool("yes") is True
    assert _as_bool("true") is True
    assert _as_bool("1") is True
    assert _as_bool("no") is False
    assert _as_bool(np.nan) is False


def test_load_stage1_normality_map_reads_csv_flags(tmp_path):
    csv_path = tmp_path / "stage1.csv"

    df = pd.DataFrame({
        "feature": ["word_count", "semantic_coherence"],
        "approx_normal_depressed": ["true", "false"],
        "approx_normal_normal": ["1", "0"],
    })
    df.to_csv(csv_path, index=False)

    out = load_stage1_normality_map(csv_path)

    assert out["word_count"]["approx_normal_depressed"] is True
    assert out["word_count"]["approx_normal_normal"] is True
    assert out["word_count"]["both_approx_normal"] is True

    assert out["semantic_coherence"]["approx_normal_depressed"] is False
    assert out["semantic_coherence"]["approx_normal_normal"] is False
    assert out["semantic_coherence"]["both_approx_normal"] is False


def test_normalise_context_value_handles_list_dict_nan_and_scalar():
    assert normalise_context_value([]) == "UNKNOWN"
    assert normalise_context_value(["a", "b"]) == "a|b"
    assert normalise_context_value({}) == "UNKNOWN"
    assert normalise_context_value({"b": 2, "a": 1}) == json.dumps(
        {"a": 1, "b": 2},
        ensure_ascii=False,
        sort_keys=True,
    )
    assert normalise_context_value(np.nan) == "UNKNOWN"
    assert normalise_context_value(123) == "123"


def test_column_is_numeric_like_detects_numeric_strings():
    s1 = pd.Series(["1", "2", "3.5"])
    s2 = pd.Series(["1", "x", "3"])
    s3 = pd.Series([None, np.nan])

    assert column_is_numeric_like(s1)
    assert not column_is_numeric_like(s2)
    assert not column_is_numeric_like(s3)


def test_get_requested_covariate_names_in_all_covariates_mode(monkeypatch):
    monkeypatch.setattr(cs, "CONTEXT_COVARIATE", "covariates")

    df = pd.DataFrame({
        "group": ["depressed", "normal"],
        "group_binary": [1, 0],
        "word_count": [100, 120],   # feature, should be excluded
        "age": [22, 24],
        "gender": ["f", "m"],
    })

    out = get_requested_covariate_names(df)

    assert set(out) == {"age", "gender"}


def test_get_requested_covariate_names_in_single_covariate_mode(monkeypatch):
    monkeypatch.setattr(cs, "CONTEXT_COVARIATE", "gender")

    df = pd.DataFrame({
        "group": ["depressed", "normal"],
        "group_binary": [1, 0],
        "gender": ["f", "m"],
        "age": [22, 24],
    })

    out = get_requested_covariate_names(df)

    assert out == ["gender"]


def test_prepare_model_covariates_builds_numeric_and_categorical_terms():
    df = pd.DataFrame({
        "age": ["20", "21", "22", "23"],
        "subreddit": ["a", "a", "b", "c"],
    })

    out_df, cov_cols, cov_terms = prepare_model_covariates(
        df,
        ["age", "subreddit"],
        top_n_levels=1,
    )

    assert cov_cols == ["age", "subreddit__major"]
    assert cov_terms == ['Q("age")', 'C(Q("subreddit__major"))']

    assert pd.api.types.is_numeric_dtype(out_df["age"])
    assert "subreddit__major" in out_df.columns
    assert set(out_df["subreddit__major"]) == {"a", "OTHER"}


def test_build_model_formula_includes_group_binary_and_covariates():
    formula = build_model_formula(
        "word_count",
        ['Q("age")', 'C(Q("subreddit__major"))'],
    )

    assert formula == 'Q("word_count") ~ group_binary + Q("age") + C(Q("subreddit__major"))'


def test_validate_feature_data_returns_clean_subset(monkeypatch):
    monkeypatch.setattr(cs, "MIN_ROWS_PER_MODEL", 4)
    monkeypatch.setattr(cs, "MIN_ROWS_PER_GROUP", 2)

    df = pd.DataFrame({
        "group": ["depressed", "depressed", "normal", "normal"],
        "group_binary": [1, 1, 0, 0],
        "word_count": [100, 120, 80, 95],
        "age": [22, 23, 24, 25],
    })

    sub = validate_feature_data(df, "word_count", ["age"])

    assert sub is not None
    assert len(sub) == 4
    assert list(sub.columns) == ["group", "group_binary", "word_count", "age"]


def test_validate_feature_data_returns_none_for_constant_feature(monkeypatch):
    monkeypatch.setattr(cs, "MIN_ROWS_PER_MODEL", 4)
    monkeypatch.setattr(cs, "MIN_ROWS_PER_GROUP", 2)

    df = pd.DataFrame({
        "group": ["depressed", "depressed", "normal", "normal"],
        "group_binary": [1, 1, 0, 0],
        "word_count": [100, 100, 100, 100],
    })

    sub = validate_feature_data(df, "word_count", [])

    assert sub is None


def test_gamma_suitability_report_accepts_positive_right_skewed_feature(monkeypatch):
    monkeypatch.setattr(cs, "MIN_ROWS_PER_MODEL", 4)

    sub = pd.DataFrame({
        "word_count": [1, 2, 3, 10, 30, 80],
    })

    report = gamma_suitability_report(
        sub=sub,
        feature="word_count",
        stage1_normality_map={},
    )

    assert report["gamma_candidate_feature"] is True
    assert report["strictly_positive"] is True
    assert report["non_constant"] is True
    assert report["gamma_ok"] is True
    assert report["gamma_reason"] == "ok"


def test_gamma_suitability_report_rejects_non_positive_values(monkeypatch):
    monkeypatch.setattr(cs, "MIN_ROWS_PER_MODEL", 4)

    sub = pd.DataFrame({
        "word_count": [0, 1, 2, 3, 4, 5],
    })

    report = gamma_suitability_report(
        sub=sub,
        feature="word_count",
        stage1_normality_map={},
    )

    assert report["gamma_ok"] is False
    assert report["gamma_reason"] == "non_positive_values_present"


def test_fit_feature_model_uses_forced_ols_branch(monkeypatch):
    monkeypatch.setattr(cs, "MIN_ROWS_PER_MODEL", 4)
    monkeypatch.setattr(cs, "MIN_ROWS_PER_GROUP", 2)

    df = pd.DataFrame({
        "group": ["depressed", "depressed", "depressed", "normal", "normal", "normal"],
        "group_binary": [1, 1, 1, 0, 0, 0],
        "semantic_coherence": [0.42, 0.51, 0.47, 0.21, 0.28, 0.31],
    })

    result = fit_feature_model(
        df=df,
        feature="semantic_coherence",
        covariate_cols=[],
        covariate_terms=[],
        stage1_normality_map={},
    )

    assert result is not None
    assert result["status"] == "ok"
    assert result["model_type"] == "ols_hc3"
    assert result["selection_reason"] == "force_ols_feature"