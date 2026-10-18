import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests



DEPRESSION_JSON = "depression_with_features_with_rates.json"
NORMAL_JSON = "normal_with_features_with_rates.json"

OUTPUT_DIR = Path("covariate_sensitivity_results")
OUTPUT_DIR.mkdir(exist_ok=True)


STAGE1_SUMMARY_CSV = Path("feature_visual_analysis") / "feature_group_summary.csv"

MIN_ROWS_PER_MODEL = 20
MIN_ROWS_PER_GROUP = 5

# set it to the variable that holds 'normal / depression'
GROUP_COVARIATE = "group"

# Two supported modes:
# 1) CONTEXT_COVARIATE = "covariates" / "covariate"
#    -> use all available covariate fields
# 2) CONTEXT_COVARIATE = a specific field name (e.g. "gender", "subreddits", "age")
#    -> use only that field
CONTEXT_COVARIATE = "covariates"

ALL_COVARIATES_MODE_KEYS = {"covariate", "covariates"}

 
TOP_N_CONTEXT_LEVELS = 20

 
SKEWNESS_MIN_FOR_GAMMA = 0.5
NEAR_ZERO_PROP_MAX = 0.25
NEAR_ZERO_RELATIVE_TO_MEDIAN = 0.05
UNIT_INTERVAL_EPS = 1e-9
 
POSITIVE_GROUP_LABEL = "depressed"
NEGATIVE_GROUP_LABEL = "normal"


# =========================
# Features
# =========================
FEATURES = [
    "word_count",
    "sentence_count",
    "mean_sentence_length",
    "semantic_coherence",
    "maximum_similarity",

    "definite_articles_per_100_words",
    "indefinite_articles_per_100_words",
    "pronouns_per_100_words",
    "first_person_singular_per_100_words",
    "first_person_plural_per_100_words",
    "second_person_per_100_words",
    "third_person_per_100_words",

    "noun_overlap",
    "pronoun_overlap",
    "content_overlap",

    "LCC",
    "LSC",
    "number_of_edges",
    "degree_average",
    "degree_std",
    "LCCr",
    "LSCr",

    "ambiguous_pronoun_count_per_100_words",
]


GAMMA_CANDIDATE_FEATURES = {
    "word_count",
    "sentence_count",
    "mean_sentence_length",

    "definite_articles_per_100_words",
    "indefinite_articles_per_100_words",
    "pronouns_per_100_words",
    "first_person_singular_per_100_words",
    "first_person_plural_per_100_words",
    "second_person_per_100_words",
    "third_person_per_100_words",
    "ambiguous_pronoun_count_per_100_words",

    "LCC",
    "LSC",
    "number_of_edges",
    "degree_average",
    "degree_std",
}


FORCE_OLS_FEATURES = {
    "semantic_coherence",
    "maximum_similarity",
    "noun_overlap",
    "pronoun_overlap",
    "content_overlap",
    "LCCr",
    "LSCr",
}



def _as_bool(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"true", "1", "yes"}


def is_all_covariates_mode() -> bool:
    return str(CONTEXT_COVARIATE).strip().lower() in ALL_COVARIATES_MODE_KEYS


def load_stage1_normality_map(csv_path: Path) -> dict:
    if not csv_path.exists():
        print(f"[Warning] Stage 1 summary CSV not found: {csv_path}")
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Warning] Failed to read Stage 1 summary CSV: {e}")
        return {}

    required_cols = {
        "feature",
        "approx_normal_depressed",
        "approx_normal_normal",
    }
    if not required_cols.issubset(df.columns):
        print("[Warning] Stage 1 summary CSV missing required columns; normality map disabled.")
        return {}

    out = {}
    for _, row in df.iterrows():
        feat = row["feature"]
        dep_norm = _as_bool(row["approx_normal_depressed"])
        con_norm = _as_bool(row["approx_normal_normal"])
        out[str(feat)] = {
            "approx_normal_depressed": dep_norm,
            "approx_normal_normal": con_norm,
            "both_approx_normal": dep_norm and con_norm,
        }
    return out


def load_group_json(json_path: str, fallback_group_label: str | None = None) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        row = {}
 
        row["id"] = item.get("id")
        row["text"] = item.get("text")
 
        covariates = item.get("covariates", None)
        if isinstance(covariates, dict):
            for k, v in covariates.items():
                row[k] = v

        for k, v in item.items():
            if k not in {"features", "id", "text", "covariates"} and k not in row:
                row[k] = v

        if GROUP_COVARIATE not in row and fallback_group_label is not None:
            row[GROUP_COVARIATE] = fallback_group_label


        feature_dict = item.get("features", {})
        if isinstance(feature_dict, dict):
            for k, v in feature_dict.items():
                row[k] = v

        rows.append(row)

    return pd.DataFrame(rows)


def build_dataframe(dep_path: str, norm_path: str) -> pd.DataFrame:
    df_dep = load_group_json(dep_path, fallback_group_label=POSITIVE_GROUP_LABEL)
    df_norm = load_group_json(norm_path, fallback_group_label=NEGATIVE_GROUP_LABEL)
    df = pd.concat([df_dep, df_norm], ignore_index=True)

    if GROUP_COVARIATE not in df.columns:
        raise ValueError(f"Required group covariate '{GROUP_COVARIATE}' not found in input data.")

    df[GROUP_COVARIATE] = df[GROUP_COVARIATE].astype(str)
    df["group_binary"] = (df[GROUP_COVARIATE] == POSITIVE_GROUP_LABEL).astype(int)

    return df


def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def resolve_features(df: pd.DataFrame, desired_features):
    ok = []
    missing = []
    for f in desired_features:
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
            ok.append(f)
        else:
            missing.append(f)

    if missing:
        print("[Warning] Missing/non-numeric features skipped:")
        for m in missing:
            print("  -", m)

    return ok


def normalise_context_value(x):
    if isinstance(x, list):
        if len(x) == 0:
            return "UNKNOWN"
        return "|".join(map(str, x))
    if isinstance(x, dict):
        if len(x) == 0:
            return "UNKNOWN"
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    if pd.isna(x):
        return "UNKNOWN"
    return str(x)


def column_is_numeric_like(series: pd.Series) -> bool:
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    converted = pd.to_numeric(non_null, errors="coerce")
    return converted.notna().all()


def get_requested_covariate_names(df: pd.DataFrame) -> list[str]:
     
    if is_all_covariates_mode():
        excluded = {"id", "text", GROUP_COVARIATE, "group_binary", *FEATURES}
        covariate_names = [c for c in df.columns if c not in excluded]
        return covariate_names

    if CONTEXT_COVARIATE not in df.columns:
        raise ValueError(
            f"Required context covariate '{CONTEXT_COVARIATE}' not found in input data."
        )
    return [CONTEXT_COVARIATE]


def prepare_model_covariates(
    df: pd.DataFrame,
    covariate_names: list[str],
    top_n_levels: int = 20,
):
    
    df = df.copy()
    covariate_cols = []
    covariate_terms = []

    for col in covariate_names:
        if col not in df.columns:
            continue
        if col == GROUP_COVARIATE:
            continue

        series = df[col]

        # numeric covariate
        if pd.api.types.is_numeric_dtype(series) or column_is_numeric_like(series):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            covariate_cols.append(col)
            covariate_terms.append(f'Q("{col}")')
            continue

        # categorical covariate
        major_col = f"{col}__major"
        df[major_col] = df[col].apply(normalise_context_value)

        top_levels = (
            df[major_col]
            .value_counts(dropna=False)
            .head(top_n_levels)
            .index
            .tolist()
        )

        df[major_col] = np.where(
            df[major_col].isin(top_levels),
            df[major_col],
            "OTHER"
        )

        covariate_cols.append(major_col)
        covariate_terms.append(f'C(Q("{major_col}"))')

    return df, covariate_cols, covariate_terms


def build_model_formula(feature: str, covariate_terms: list[str]) -> str:
    rhs = ["group_binary"] + covariate_terms
    rhs_str = " + ".join(rhs)
    return f'Q("{feature}") ~ {rhs_str}'


def validate_feature_data(
    df: pd.DataFrame,
    feature: str,
    covariate_cols: list[str],
) -> pd.DataFrame | None:
    cols = [GROUP_COVARIATE, "group_binary", feature] + covariate_cols
    cols = [c for c in cols if c in df.columns]

    sub = df[cols].copy()
    sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
    sub = sub.dropna()

    if len(sub) < MIN_ROWS_PER_MODEL:
        return None

    group_counts = sub[GROUP_COVARIATE].value_counts()
    if POSITIVE_GROUP_LABEL not in group_counts or NEGATIVE_GROUP_LABEL not in group_counts:
        return None
    if group_counts[POSITIVE_GROUP_LABEL] < MIN_ROWS_PER_GROUP or group_counts[NEGATIVE_GROUP_LABEL] < MIN_ROWS_PER_GROUP:
        return None

    if sub[feature].nunique() < 2:
        return None

    return sub


def gamma_suitability_report(
    sub: pd.DataFrame,
    feature: str,
    stage1_normality_map: dict | None = None,
) -> dict:
    y = pd.to_numeric(sub[feature], errors="coerce").dropna().values.astype(float)

    report = {
        "feature": feature,
        "gamma_candidate_feature": feature in GAMMA_CANDIDATE_FEATURES,
        "stage1_both_approx_normal": False,
        "strictly_positive": False,
        "non_constant": False,
        "bounded_0_1_like": False,
        "skewness": np.nan,
        "right_skewed_enough": False,
        "near_zero_threshold": np.nan,
        "near_zero_prop": np.nan,
        "near_zero_ok": False,
        "gamma_ok": False,
        "gamma_reason": "",
    }

    if len(y) < MIN_ROWS_PER_MODEL:
        report["gamma_reason"] = "too_few_rows"
        return report

    if feature not in GAMMA_CANDIDATE_FEATURES:
        report["gamma_reason"] = "not_in_gamma_candidate_set"
        return report

    if stage1_normality_map is not None and feature in stage1_normality_map:
        report["stage1_both_approx_normal"] = bool(
            stage1_normality_map[feature].get("both_approx_normal", False)
        )
        if report["stage1_both_approx_normal"]:
            report["gamma_reason"] = "approximately_normal_in_stage1"
            return report

    if np.any(y <= 0):
        report["gamma_reason"] = "non_positive_values_present"
        return report
    report["strictly_positive"] = True

    if np.nanstd(y) == 0:
        report["gamma_reason"] = "constant_values"
        return report
    report["non_constant"] = True

    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    if y_min >= -UNIT_INTERVAL_EPS and y_max <= 1 + UNIT_INTERVAL_EPS:
        report["bounded_0_1_like"] = True
        report["gamma_reason"] = "bounded_unit_interval_like"
        return report

    skewness = float(pd.Series(y).skew())
    report["skewness"] = skewness
    if not np.isfinite(skewness) or skewness < SKEWNESS_MIN_FOR_GAMMA:
        report["gamma_reason"] = "not_right_skewed_enough"
        return report
    report["right_skewed_enough"] = True

    median_y = float(np.nanmedian(y))
    near_zero_threshold = max(1e-8, median_y * NEAR_ZERO_RELATIVE_TO_MEDIAN)
    near_zero_prop = float(np.mean(y <= near_zero_threshold))

    report["near_zero_threshold"] = near_zero_threshold
    report["near_zero_prop"] = near_zero_prop

    if near_zero_prop > NEAR_ZERO_PROP_MAX:
        report["gamma_reason"] = "too_much_mass_near_zero"
        return report
    report["near_zero_ok"] = True

    report["gamma_ok"] = True
    report["gamma_reason"] = "ok"
    return report


 
def fit_ols_hc3(
    df: pd.DataFrame,
    feature: str,
    covariate_cols: list[str],
    covariate_terms: list[str],
    stage1_normality_map: dict | None = None,
) -> dict | None:
    sub = validate_feature_data(df, feature, covariate_cols=covariate_cols)
    if sub is None:
        return None

    formula = build_model_formula(feature, covariate_terms)

    try:
        model = smf.ols(formula, data=sub).fit(cov_type="HC3")
    except Exception as e:
        return {
            "feature": feature,
            "model_type": "ols_hc3",
            "status": f"fit_failed: {e}",
        }

    term = "group_binary"

    try:
        ci = model.conf_int().loc[term]
        ci_low, ci_high = float(ci[0]), float(ci[1])
    except Exception:
        ci_low, ci_high = np.nan, np.nan

    stage1_both_approx_normal = False
    if stage1_normality_map is not None and feature in stage1_normality_map:
        stage1_both_approx_normal = bool(
            stage1_normality_map[feature].get("both_approx_normal", False)
        )

    return {
        "feature": feature,
        "model_type": "ols_hc3",
        "status": "ok",
        "n": int(model.nobs),
        "coef_group": float(model.params.get(term, np.nan)),
        "se_group": float(model.bse.get(term, np.nan)),
        "z_or_t_group": float(model.tvalues.get(term, np.nan)),
        "p_group": float(model.pvalues.get(term, np.nan)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "effect_interpretation": "additive",
        "lambda_or_irr": np.nan,
        "lambda_or_irr_ci_low": np.nan,
        "lambda_or_irr_ci_high": np.nan,
        "aic": float(getattr(model, "aic", np.nan)),
        "bic": float(getattr(model, "bic", np.nan)),
        "stage1_both_approx_normal": stage1_both_approx_normal,
        "gamma_reason": np.nan,
        "gamma_skewness": np.nan,
        "gamma_near_zero_prop": np.nan,
        "gamma_bounded_0_1_like": np.nan,
        "selection_reason": np.nan,
    }


def fit_gamma_glm(
    df: pd.DataFrame,
    feature: str,
    covariate_cols: list[str],
    covariate_terms: list[str],
    stage1_normality_map: dict | None = None,
) -> dict | None:
    sub = validate_feature_data(df, feature, covariate_cols=covariate_cols)
    if sub is None:
        return None

    gamma_check = gamma_suitability_report(
        sub=sub,
        feature=feature,
        stage1_normality_map=stage1_normality_map,
    )

    if not gamma_check["gamma_ok"]:
        return {
            "feature": feature,
            "model_type": "gamma_glm_log",
            "status": "incompatible_with_gamma",
            "gamma_reason": gamma_check["gamma_reason"],
            "stage1_both_approx_normal": gamma_check["stage1_both_approx_normal"],
            "gamma_skewness": gamma_check["skewness"],
            "gamma_near_zero_prop": gamma_check["near_zero_prop"],
            "gamma_bounded_0_1_like": gamma_check["bounded_0_1_like"],
            "selection_reason": np.nan,
        }

    formula = build_model_formula(feature, covariate_terms)

    try:
        model = smf.glm(
            formula=formula,
            data=sub,
            family=sm.families.Gamma(link=sm.families.links.log())
        ).fit()
    except Exception as e:
        return {
            "feature": feature,
            "model_type": "gamma_glm_log",
            "status": f"fit_failed: {e}",
            "gamma_reason": "fit_failed",
            "stage1_both_approx_normal": gamma_check["stage1_both_approx_normal"],
            "gamma_skewness": gamma_check["skewness"],
            "gamma_near_zero_prop": gamma_check["near_zero_prop"],
            "gamma_bounded_0_1_like": gamma_check["bounded_0_1_like"],
            "selection_reason": np.nan,
        }

    term = "group_binary"

    try:
        ci = model.conf_int().loc[term]
        ci_low, ci_high = float(ci[0]), float(ci[1])
    except Exception:
        ci_low, ci_high = np.nan, np.nan

    coef = float(model.params.get(term, np.nan))

    return {
        "feature": feature,
        "model_type": "gamma_glm_log",
        "status": "ok",
        "n": int(model.nobs),
        "coef_group": coef,
        "se_group": float(model.bse.get(term, np.nan)),
        "z_or_t_group": float(model.tvalues.get(term, np.nan)),
        "p_group": float(model.pvalues.get(term, np.nan)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "effect_interpretation": "multiplicative",
        "lambda_or_irr": float(np.exp(coef)) if not pd.isna(coef) else np.nan,
        "lambda_or_irr_ci_low": float(np.exp(ci_low)) if not pd.isna(ci_low) else np.nan,
        "lambda_or_irr_ci_high": float(np.exp(ci_high)) if not pd.isna(ci_high) else np.nan,
        "aic": float(getattr(model, "aic", np.nan)),
        "bic": np.nan,
        "stage1_both_approx_normal": gamma_check["stage1_both_approx_normal"],
        "gamma_reason": gamma_check["gamma_reason"],
        "gamma_skewness": gamma_check["skewness"],
        "gamma_near_zero_prop": gamma_check["near_zero_prop"],
        "gamma_bounded_0_1_like": gamma_check["bounded_0_1_like"],
        "selection_reason": np.nan,
    }


def fit_feature_model(
    df: pd.DataFrame,
    feature: str,
    covariate_cols: list[str],
    covariate_terms: list[str],
    stage1_normality_map: dict | None = None,
) -> dict | None:
    if feature in FORCE_OLS_FEATURES:
        result = fit_ols_hc3(
            df,
            feature,
            covariate_cols=covariate_cols,
            covariate_terms=covariate_terms,
            stage1_normality_map=stage1_normality_map,
        )
        if result is not None:
            result["selection_reason"] = "force_ols_feature"
        return result

    gamma_result = fit_gamma_glm(
        df,
        feature,
        covariate_cols=covariate_cols,
        covariate_terms=covariate_terms,
        stage1_normality_map=stage1_normality_map,
    )

    if gamma_result is not None and gamma_result.get("status") == "ok":
        gamma_result["selection_reason"] = "gamma_checks_passed"
        return gamma_result

    ols_result = fit_ols_hc3(
        df,
        feature,
        covariate_cols=covariate_cols,
        covariate_terms=covariate_terms,
        stage1_normality_map=stage1_normality_map,
    )
    if ols_result is not None:
        if gamma_result is not None:
            ols_result["selection_reason"] = f"fallback_to_ols:{gamma_result.get('gamma_reason', 'unknown')}"
        else:
            ols_result["selection_reason"] = "default_ols"
    return ols_result


 
def main():
    warnings.filterwarnings("ignore")

    stage1_normality_map = load_stage1_normality_map(STAGE1_SUMMARY_CSV)

    df = build_dataframe(DEPRESSION_JSON, NORMAL_JSON)
    df = coerce_numeric(df, FEATURES)

    features = resolve_features(df, FEATURES)
    if not features:
        print("No usable features found.")
        return

    requested_covariates = get_requested_covariate_names(df)
    df, covariate_cols, covariate_terms = prepare_model_covariates(
        df,
        requested_covariates,
        top_n_levels=TOP_N_CONTEXT_LEVELS,
    )

    print(f"Total records: {len(df)}")
    print(f"Group covariate: {GROUP_COVARIATE}")
    print(f"Context setting: {CONTEXT_COVARIATE}")
    print(f"Resolved covariates: {requested_covariates}")
    print(f"Model covariate columns: {covariate_cols}")
    print("Features used:")
    for f in features:
        print("  -", f)

    if stage1_normality_map:
        print(f"\n[Info] Loaded Stage 1 normality summary from: {STAGE1_SUMMARY_CSV.resolve()}")
    else:
        print("\n[Warning] Stage 1 normality summary not available; model selection will rely only on Stage 4 checks.")

    results = []
    skipped_features = []

    for feature in features:
        result = fit_feature_model(
            df,
            feature,
            covariate_cols=covariate_cols,
            covariate_terms=covariate_terms,
            stage1_normality_map=stage1_normality_map,
        )
        if result is not None:
            results.append(result)
        else:
            skipped_features.append(feature)

    if skipped_features:
        print("\n[Info] Skipped features:")
        for f in skipped_features:
            print("  -", f)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No model results produced.")
        return

    ok_mask = (results_df["status"] == "ok") & results_df["p_group"].notna()
    results_df["q_group"] = np.nan
    if ok_mask.sum() > 0:
        _, qvals, _, _ = multipletests(results_df.loc[ok_mask, "p_group"], method="fdr_bh")
        results_df.loc[ok_mask, "q_group"] = qvals

    results_df = results_df.sort_values(
        by=["q_group", "p_group"],
        ascending=[True, True],
        na_position="last"
    ).reset_index(drop=True)

    out_csv = OUTPUT_DIR / "covariate_sensitivity_results.csv"
    results_df.to_csv(
        out_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDone.")
    print(f"Saved: {out_csv.resolve()}")

    print("\n[Info] Model types used:")
    print(results_df["model_type"].value_counts(dropna=False))

    if "selection_reason" in results_df.columns:
        print("\n[Info] Selection reasons:")
        print(results_df["selection_reason"].value_counts(dropna=False))

    if "status" in results_df.columns:
        print("\n[Info] Status summary:")
        print(results_df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()