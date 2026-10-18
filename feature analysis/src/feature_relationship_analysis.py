import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

 
DEPRESSION_JSON = "depression_with_features_with_rates.json"
NORMAL_JSON = "normal_with_features_with_rates.json"

OUTPUT_DIR = Path("feature_relationship_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# set it to the variable that holds 'normal/depression'
GROUP_COVARIATE = "group"

# set it to the confounder variables
CONTEXT_COVARIATE = "covariates"
 
# features to be analysed
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

FEATURE_LABELS = {
    "word_count": "No. words",
    "sentence_count": "No. sent.s",
    "mean_sentence_length": "Mean sent. length",
    "semantic_coherence": "Coherence",
    "maximum_similarity": "Max. similarity",

    "definite_articles_per_100_words": "Def. articles /100w",
    "indefinite_articles_per_100_words": "Indef. articles /100w",
    "pronouns_per_100_words": "Pronouns /100w",
    "first_person_singular_per_100_words": "1st pers. sing. /100w",
    "first_person_plural_per_100_words": "1st pers. plur. /100w",
    "second_person_per_100_words": "2nd person /100w",
    "third_person_per_100_words": "3rd person /100w",

    "noun_overlap": "Noun overlap",
    "pronoun_overlap": "Pronoun overlap",
    "content_overlap": "Content overlap",

    "LCC": "LCC",
    "LSC": "LSC",
    "number_of_edges": "No. edges",
    "degree_average": "Degree avg.",
    "degree_std": "Degree std.",
    "LCCr": "LCCr",
    "LSCr": "LSCr",

    "ambiguous_pronoun_count_per_100_words": "Ambig. pron. /100w",
}

P_THRESHOLD = 0.01
MIN_ROWS = 20

 
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
            # schema A: nested covariates
            for k, v in covariates.items():
                row[k] = v
        else:
            # schema B: top-level covariates
            for k, v in item.items():
                if k not in {"features", "id", "text"}:
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
    df_dep = load_group_json(dep_path, "depressed")
    df_norm = load_group_json(norm_path, "normal")
    df = pd.concat([df_dep, df_norm], ignore_index=True)

    df["group"] = df["group"].astype(str)
    df["group_binary"] = (df["group"] == "depressed").astype(int)

    return df


def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def resolve_features(df: pd.DataFrame, features):
    ok = []
    missing = []
    for f in features:
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
            ok.append(f)
        else:
            missing.append(f)

    if missing:
        print("[Warning] Missing/non-numeric features skipped:")
        for m in missing:
            print("  -", m)

    return ok





# =========================
# pairwise linear regression
# feature_y ~ feature_x + group
# =========================
def pairwise_regression_matrix(df: pd.DataFrame, features: list[str]):
    n = len(features)

    t_mat = pd.DataFrame(np.nan, index=features, columns=features)
    p_mat = pd.DataFrame(np.nan, index=features, columns=features)
    coef_mat = pd.DataFrame(np.nan, index=features, columns=features)

    long_rows = []

    for i, fx in enumerate(features):
        for j, fy in enumerate(features):
            if i == j:
                t_mat.loc[fy, fx] = np.nan
                p_mat.loc[fy, fx] = np.nan
                coef_mat.loc[fy, fx] = np.nan
                continue

            sub = df[[fx, fy, "group_binary"]].copy()
            sub[fx] = pd.to_numeric(sub[fx], errors="coerce")
            sub[fy] = pd.to_numeric(sub[fy], errors="coerce")
            sub = sub.dropna()

            if len(sub) < MIN_ROWS:
                continue
            if sub[fx].nunique() < 2 or sub[fy].nunique() < 2:
                continue
            if sub["group_binary"].nunique() < 2:
                continue

            formula = f'Q("{fy}") ~ Q("{fx}") + group_binary'

            try:
                model = smf.ols(formula, data=sub).fit()
                term = f'Q("{fx}")'

                coef = float(model.params.get(term, np.nan))
                tval = float(model.tvalues.get(term, np.nan))
                pval = float(model.pvalues.get(term, np.nan))

                coef_mat.loc[fy, fx] = coef
                t_mat.loc[fy, fx] = tval
                p_mat.loc[fy, fx] = pval

                long_rows.append({
                    "feature_x": fx,
                    "feature_y": fy,
                    "coef_x": coef,
                    "t_x": tval,
                    "p_x": pval,
                    "n": int(model.nobs),
                    "r_squared": float(model.rsquared),
                })

            except Exception as e:
                long_rows.append({
                    "feature_x": fx,
                    "feature_y": fy,
                    "coef_x": np.nan,
                    "t_x": np.nan,
                    "p_x": np.nan,
                    "n": len(sub),
                    "r_squared": np.nan,
                    "error": str(e),
                })

    long_df = pd.DataFrame(long_rows)
    return coef_mat, t_mat, p_mat, long_df


# =========================
# draw heat map
# =========================
def plot_heatmap(matrix: pd.DataFrame, title: str, output_path: Path, cmap="coolwarm", vlim=(-10, 10)):
    labels = [FEATURE_LABELS.get(c, c) for c in matrix.columns]

    fig, ax = plt.subplots(figsize=(12, 10))
    data = matrix.values.astype(float)

    vmin, vmax = vlim
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_title(title, fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("T-statistic", rotation=90)

    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_significant_heatmap(t_mat: pd.DataFrame, p_mat: pd.DataFrame, p_thresh: float, output_path: Path, vlim=(-10, 10)):
    sig_t = t_mat.copy()
    sig_t[p_mat >= p_thresh] = np.nan

    labels = [FEATURE_LABELS.get(c, c) for c in sig_t.columns]

    fig, ax = plt.subplots(figsize=(12, 10))
    data = sig_t.values.astype(float)

    vmin, vmax = vlim

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_title(f"T-statistic heat map (only P < {p_thresh})", fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("T-statistic", rotation=90)

    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)



def main():
    df = build_dataframe(DEPRESSION_JSON, NORMAL_JSON)
    df = coerce_numeric(df, FEATURES)

    features = resolve_features(df, FEATURES)
    if not features:
        print("No usable features found.")
        return


    df_model = df.copy()
    print(f"[Info] Using data: {len(df_model)} rows")

    coef_mat, t_mat, p_mat, long_df = pairwise_regression_matrix(df_model, features)

    
    coef_mat.to_csv(OUTPUT_DIR / "pairwise_coef_matrix.csv", encoding="utf-8-sig")
    t_mat.to_csv(OUTPUT_DIR / "pairwise_t_matrix.csv", encoding="utf-8-sig")
    p_mat.to_csv(OUTPUT_DIR / "pairwise_p_matrix.csv", encoding="utf-8-sig")
    long_df.to_csv(OUTPUT_DIR / "pairwise_regression_long.csv", index=False, encoding="utf-8-sig")

    
    plot_heatmap(
        t_mat,
        title="Relationships between NLP measures (T-statistics)",
        output_path=OUTPUT_DIR / "heatmap_t_statistics.png",
    )

    plot_significant_heatmap(
        t_mat,
        p_mat,
        p_thresh=P_THRESHOLD,
        output_path=OUTPUT_DIR / f"heatmap_t_statistics_p_lt_{str(P_THRESHOLD).replace('.', '_')}.png",
    )

    print("Done.")
    print(f"Saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()