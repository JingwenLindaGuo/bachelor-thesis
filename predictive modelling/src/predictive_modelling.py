import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# ===== optional XGBoost =====
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# =========================
# Configuration
# =========================
DEPRESSION_JSON = "depression_user_level_with_rates.json"
NORMAL_JSON = "normal_user_level_with_rates.json"

OUTPUT_DIR = Path("predictive_modelling_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 2025

# Prefer rate-normalised features if available; otherwise fall back to raw counts.
FEATURE_SPECS = {
    "lexical": [
        ("definite_articles_per_100_words", "definite_articles"),
        ("indefinite_articles_per_100_words", "indefinite_articles"),
        ("pronouns_per_100_words", "pronouns"),
        ("first_person_singular_per_100_words", "first_person_singular"),
        ("first_person_plural_per_100_words", "first_person_plural"),
        ("second_person_per_100_words", "second_person"),
        ("third_person_per_100_words", "third_person"),
    ],
    "discourse_coherence": [
        ("semantic_coherence",),
        ("maximum_similarity", "repetition"),
        (
            "ambiguous_pronoun_usage_per_100_words",
            "ambiguous_pronoun_usage",
            "ambiguous_pronoun_count_per_100_words",
            "ambiguous_pronoun_count",
            "ambiguous_pronouns",
        ),
        ("noun_overlap",),
        ("pronoun_overlap",),
        ("content_overlap",),
    ],
    "syntactic": [
        ("mean_sentence_length", "average_sentence_length"),
        ("average_subtree_depth", "avg_subtree_depth"),
        ("maximum_subtree_depth", "max_subtree_depth"),
        ("vp_count_per_sentence", "number_of_verb_phrases", "vp_count", "num_vp", "VP_count"),
        ("vp_length", "mean_vp_length", "average_vp_length"),
        ("sbar_count_per_sentence", "number_of_subordinate_clauses", "sbar_count", "num_sbar", "SBAR_count"),
        ("sbar_length", "mean_sbar_length", "average_sbar_length"),
    ],
    "speech_graph": [
        ("LCC",),
        ("LSC",),
        ("LCCr",),
        ("LSCr",),
        ("number_of_edges",),
        ("degree_average",),
        ("degree_std",),
    ],
}

# IMPORTANT: update this set for the current input schema.
# It should include all non-feature columns that must be excluded from the
# predictive feature matrix, such as identifiers, raw metadata, labels, and
# any potential leakage variables.
EXCLUDE_COLUMNS = {
    "author",
    "id",
    "text",
    "created_utc",
    "subreddit",
    "subreddit_major",
    "subreddits",
    "ids",
    "evidence",
    "llm_label",
    "votes",
    "label",
    "group",
    "first_created_utc",
    "last_created_utc",
}


# =========================
# Data loading
# =========================
def load_json_records(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "records", "items", "users", "authors"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Unsupported JSON structure in {path}")


def load_group_df(path: str, label: int) -> pd.DataFrame:
    records = load_json_records(path)
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No records found in {path}")

    if "features" in df.columns:
        features_df = pd.json_normalize(df["features"])
        df = pd.concat([df.drop(columns=["features"]), features_df], axis=1)

    df["label"] = label
    return df


# =========================
# Feature handling
# =========================
def choose_first_existing(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def resolve_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    resolved = {}

    for group_name, specs in FEATURE_SPECS.items():
        cols = []
        for candidates in specs:
            chosen = choose_first_existing(df, candidates)
            if chosen is not None and chosen not in cols:
                cols.append(chosen)
        resolved[group_name] = cols

    all_cols = []
    for group_name in ["lexical", "discourse_coherence", "syntactic", "speech_graph"]:
        for c in resolved[group_name]:
            if c not in all_cols:
                all_cols.append(c)

    resolved["all_features"] = all_cols
    return resolved


def ensure_numeric_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def print_feature_groups(feature_groups: Dict[str, List[str]]):
    print("\n[Feature groups being used]")
    for name, cols in feature_groups.items():
        print(f"  {name}: {len(cols)} features")
        print(f"    {cols}")


# =========================
# Models
# =========================
def build_models():
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )),
            ]
        ),
        "linear_svm": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )),
            ]
        ),
    }

    if HAS_XGBOOST:
        models["xgboost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]
        )
    else:
        print("\n[Warning] xgboost is not installed, so XGBoost will be skipped.")

    return models


# =========================
# CV evaluation
# =========================
def evaluate_models(
    df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    models: Dict[str, Pipeline],
):
    y = df["label"].astype(int).values

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    fold_rows = []
    summary_rows = []

    for feature_set_name, cols in feature_groups.items():
        if len(cols) == 0:
            print(f"\n[Skip] {feature_set_name}: no matching columns found")
            continue

        X = ensure_numeric_features(df, cols)

        print(f"\n[Running] feature_set = {feature_set_name} | n_features = {len(cols)}")

        for model_name, model in models.items():
            print(f"  -> model = {model_name}")

            scores = cross_validate(
                estimator=model,
                X=X,
                y=y,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=1,
            )

            n_folds = len(scores["test_f1"])
            for i in range(n_folds):
                fold_rows.append({
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "fold": i + 1,
                    "precision": scores["test_precision"][i],
                    "recall": scores["test_recall"][i],
                    "f1": scores["test_f1"][i],
                    "roc_auc": scores["test_roc_auc"][i],
                    "n_features": len(cols),
                    "features_used": ", ".join(cols),
                })

            summary_rows.append({
                "feature_set": feature_set_name,
                "model": model_name,
                "n_features": len(cols),
                "features_used": ", ".join(cols),

                "precision_mean": np.mean(scores["test_precision"]),
                "precision_std": np.std(scores["test_precision"], ddof=1),

                "recall_mean": np.mean(scores["test_recall"]),
                "recall_std": np.std(scores["test_recall"], ddof=1),

                "f1_mean": np.mean(scores["test_f1"]),
                "f1_std": np.std(scores["test_f1"], ddof=1),

                "roc_auc_mean": np.mean(scores["test_roc_auc"]),
                "roc_auc_std": np.std(scores["test_roc_auc"], ddof=1),
            })

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)

    if not fold_df.empty:
        fold_df.to_csv(OUTPUT_DIR / "cv_fold_results.csv", index=False)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["feature_set", "f1_mean", "roc_auc_mean"],
            ascending=[True, False, False]
        )
        summary_df.to_csv(OUTPUT_DIR / "cv_summary_results.csv", index=False)

    return fold_df, summary_df


# =========================
# Importance export
# =========================
def export_tree_importances(
    df: pd.DataFrame,
    models: Dict[str, Pipeline],
    feature_groups: Dict[str, List[str]],
    feature_set_name: str = "all_features",
):
    cols = feature_groups.get(feature_set_name, [])
    if not cols:
        print(f"\n[Skip importance] No columns found for {feature_set_name}")
        return

    X = ensure_numeric_features(df, cols)
    y = df["label"].astype(int).values

    for model_name in ["random_forest", "xgboost"]:
        if model_name not in models:
            continue

        model = models[model_name]
        model.fit(X, y)

        clf = model.named_steps["model"]
        if not hasattr(clf, "feature_importances_"):
            continue

        imp = pd.DataFrame({
            "feature": cols,
            "importance": clf.feature_importances_
        }).sort_values(by="importance", ascending=False)

        out_path = OUTPUT_DIR / f"{model_name}_{feature_set_name}_feature_importance.csv"
        imp.to_csv(out_path, index=False)
        print(f"[Saved] {out_path}")


def export_logistic_coefficients(
    df: pd.DataFrame,
    models: Dict[str, Pipeline],
    feature_groups: Dict[str, List[str]],
    feature_set_name: str = "all_features",
):
    if "logistic_regression" not in models:
        return

    cols = feature_groups.get(feature_set_name, [])
    if not cols:
        return

    X = ensure_numeric_features(df, cols)
    y = df["label"].astype(int).values

    model = models["logistic_regression"]
    model.fit(X, y)

    clf = model.named_steps["model"]
    coefs = clf.coef_.ravel()

    coef_df = pd.DataFrame({
        "feature": cols,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values(by="abs_coefficient", ascending=False)

    out_path = OUTPUT_DIR / f"logistic_regression_{feature_set_name}_coefficients.csv"
    coef_df.to_csv(out_path, index=False)
    print(f"[Saved] {out_path}")


# =========================
# Main
# =========================
def main():
    print("[Start] Loading data...")
    dep_df = load_group_df(DEPRESSION_JSON, label=1)
    ctl_df = load_group_df(NORMAL_JSON, label=0)

    df = pd.concat([dep_df, ctl_df], ignore_index=True)

    print(f"[Loaded] depressed = {len(dep_df)}, control = {len(ctl_df)}, total = {len(df)}")

    keep_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    df = df[keep_cols + ["label"]]

    feature_groups = resolve_feature_groups(df)
    print_feature_groups(feature_groups)

    models = build_models()

    print("\n[Running CV evaluation...]")
    fold_df, summary_df = evaluate_models(df, feature_groups, models)

    print("\n[Top summary rows by F1]")
    if not summary_df.empty:
        print(summary_df[[
            "feature_set", "model",
            "precision_mean", "recall_mean",
            "f1_mean", "roc_auc_mean"
        ]].head(20).to_string(index=False))
    else:
        print("No summary results generated.")

    print("\n[Exporting feature importance / coefficients on full data...]")
    export_tree_importances(df, models, feature_groups, feature_set_name="all_features")
    export_logistic_coefficients(df, models, feature_groups, feature_set_name="all_features")

    print(f"\n[Done] Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()