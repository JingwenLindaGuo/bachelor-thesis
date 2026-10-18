import json
import numpy as np
import pandas as pd
import pytest

import src.predictive_modelling as pm
from src.predictive_modelling import (
    load_json_records,
    load_group_df,
    choose_first_existing,
    resolve_feature_groups,
    ensure_numeric_features,
    build_models,
    evaluate_models,
    export_logistic_coefficients,
)


def test_load_json_records_accepts_list(tmp_path):
    path = tmp_path / "data.json"
    payload = [{"id": "1"}, {"id": "2"}]
    path.write_text(json.dumps(payload), encoding="utf-8")

    out = load_json_records(str(path))

    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["id"] == "1"


def test_load_json_records_accepts_wrapped_records_dict(tmp_path):
    path = tmp_path / "data.json"
    payload = {"records": [{"id": "1"}, {"id": "2"}]}
    path.write_text(json.dumps(payload), encoding="utf-8")

    out = load_json_records(str(path))

    assert isinstance(out, list)
    assert len(out) == 2
    assert out[1]["id"] == "2"


def test_load_json_records_rejects_invalid_structure(tmp_path):
    path = tmp_path / "data.json"
    payload = {"unexpected": "value"}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported JSON structure"):
        load_json_records(str(path))


def test_load_group_df_flattens_features_and_adds_label(tmp_path):
    path = tmp_path / "group.json"
    payload = [
        {
            "id": "001",
            "author": "u1",
            "features": {
                "semantic_coherence": 0.33,
                "LCC": 10
            }
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    df = load_group_df(str(path), label=1)

    assert len(df) == 1
    assert "semantic_coherence" in df.columns
    assert "LCC" in df.columns
    assert "features" not in df.columns
    assert df.loc[0, "label"] == 1


def test_choose_first_existing_returns_first_match():
    df = pd.DataFrame({
        "b": [1, 2],
        "c": [3, 4],
    })

    out = choose_first_existing(df, ("a", "b", "c"))

    assert out == "b"


def test_choose_first_existing_returns_none_when_missing():
    df = pd.DataFrame({"x": [1, 2]})

    out = choose_first_existing(df, ("a", "b"))

    assert out is None


def test_resolve_feature_groups_uses_available_columns():
    df = pd.DataFrame({
        "definite_articles_per_100_words": [1, 2],
        "semantic_coherence": [0.2, 0.3],
        "maximum_similarity": [0.7, 0.8],
        "LCC": [10, 12],
        "degree_average": [2.1, 2.2],
    })

    out = resolve_feature_groups(df)

    assert "lexical" in out
    assert "discourse_coherence" in out
    assert "speech_graph" in out
    assert "all_features" in out

    assert "definite_articles_per_100_words" in out["lexical"]
    assert "semantic_coherence" in out["discourse_coherence"]
    assert "LCC" in out["speech_graph"]

    assert len(out["all_features"]) == len(set(out["all_features"]))


def test_ensure_numeric_features_coerces_invalid_values_to_nan():
    df = pd.DataFrame({
        "f1": ["1", "2", "bad"],
        "f2": ["0.1", "0.2", "0.3"],
    })

    X = ensure_numeric_features(df, ["f1", "f2"])

    assert pd.api.types.is_numeric_dtype(X["f1"])
    assert pd.api.types.is_numeric_dtype(X["f2"])
    assert np.isnan(X.loc[2, "f1"])


def test_build_models_returns_expected_model_keys():
    models = build_models()

    assert "logistic_regression" in models
    assert "linear_svm" in models
    assert "random_forest" in models


def test_evaluate_models_returns_fold_and_summary_dfs(monkeypatch):
    df = pd.DataFrame({
        "label": [0, 1, 0, 1],
        "f1": [1.0, 2.0, 1.5, 2.5],
        "f2": [0.1, 0.2, 0.3, 0.4],
    })

    feature_groups = {
        "set_a": ["f1", "f2"],
        "empty_set": [],
    }

    models = {"dummy_model": object()}

    def fake_cross_validate(estimator, X, y, cv, scoring, return_train_score, n_jobs):
        return {
            "test_precision": np.array([0.7, 0.8]),
            "test_recall": np.array([0.6, 0.9]),
            "test_f1": np.array([0.65, 0.85]),
            "test_roc_auc": np.array([0.75, 0.95]),
        }

    monkeypatch.setattr(pm, "cross_validate", fake_cross_validate)

    fold_df, summary_df = evaluate_models(df, feature_groups, models)

    assert not fold_df.empty
    assert not summary_df.empty

    assert set(["feature_set", "model", "fold", "precision", "recall", "f1", "roc_auc"]).issubset(fold_df.columns)
    assert set(["feature_set", "model", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean"]).issubset(summary_df.columns)

    assert set(summary_df["feature_set"]) == {"set_a"}
    assert set(summary_df["model"]) == {"dummy_model"}


def test_export_logistic_coefficients_writes_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)

    class DummyClassifier:
        coef_ = np.array([[0.5, -1.2]])

    class DummyPipeline:
        named_steps = {"model": DummyClassifier()}

        def fit(self, X, y):
            return self

    df = pd.DataFrame({
        "label": [0, 1, 0, 1],
        "f1": [1.0, 2.0, 1.5, 2.5],
        "f2": [0.1, 0.2, 0.3, 0.4],
    })

    models = {"logistic_regression": DummyPipeline()}
    feature_groups = {"all_features": ["f1", "f2"]}

    export_logistic_coefficients(df, models, feature_groups, feature_set_name="all_features")

    out = tmp_path / "logistic_regression_all_features_coefficients.csv"
    assert out.exists()

    coef_df = pd.read_csv(out)
    assert set(["feature", "coefficient", "abs_coefficient"]).issubset(coef_df.columns)
    assert len(coef_df) == 2