import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
 
NORMALIZED_COUNT_FEATURE_MAP = {
    "definite_articles": "definite_articles_per_100_words",
    "indefinite_articles": "indefinite_articles_per_100_words",
    "pronouns": "pronouns_per_100_words",
    "first_person_singular": "first_person_singular_per_100_words",
    "first_person_plural": "first_person_plural_per_100_words",
    "second_person": "second_person_per_100_words",
    "third_person": "third_person_per_100_words",
    "ambiguous_pronoun_count": "ambiguous_pronoun_count_per_100_words",
}

COUNT_FEATURES_TO_NORMALIZE = list(NORMALIZED_COUNT_FEATURE_MAP.values())


 
DEPRESSION_JSON = "depression_with_features_with_rates.json"
NORMAL_JSON = "normal_with_features_with_rates.json"

OUTPUT_DIR = Path("feature_visual_analysis")
SPIDER_PNG_1 = OUTPUT_DIR / "group_spider_plot_part1.png"
SPIDER_PNG_2 = OUTPUT_DIR / "group_spider_plot_part2.png"

BOXPLOT_PNG_1 = OUTPUT_DIR / "group_boxplots_part1.png"
BOXPLOT_PNG_2 = OUTPUT_DIR / "group_boxplots_part2.png"
SUMMARY_CSV = OUTPUT_DIR / "feature_group_summary.csv"

RANDOM_SEED = 2025
SHAPIRO_MAX_N = 5000
ALPHA_NORMALITY = 0.05



# =========================
# Feature definitions
# =========================
FEATURE_SET_1 = [
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
]

FEATURE_SET_2 = [
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

PLOT_FEATURES = FEATURE_SET_1 + FEATURE_SET_2

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

GROUP_ORDER = ["normal", "depressed"]
GROUP_DISPLAY_NAMES = {
    "normal": "Control",
    "depressed": "Depressed",
}

# which variable holds depression/normal
GROUP_COVARIATE = "group"

# which variable hold confounder varaibles
CONTEXT_COVARIATE = "covariates"

# =========================
# Data loading
# =========================
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


def build_dataframe(depression_json: str, normal_json: str) -> pd.DataFrame:
    df_dep = load_group_json(depression_json, "depressed")
    df_norm = load_group_json(normal_json, "normal")
    df = pd.concat([df_dep, df_norm], ignore_index=True)
    return df


# =========================
# Utility functions
# =========================
def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def resolve_plot_features(df: pd.DataFrame, desired_features):
    available = []
    missing = []

    for feat in desired_features:
        if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
            available.append(feat)
        else:
            missing.append(feat)

    if missing:
        print("[Warning] These features are missing or non-numeric and will be skipped:")
        for m in missing:
            print("   -", m)

    return available


def compute_group_means(df: pd.DataFrame, features):
    return df.groupby("group")[features].mean(numeric_only=True)


def compute_feature_ranges(df: pd.DataFrame, features):
    s_vec = []
    e_vec = []

    for feat in features:
        x = pd.to_numeric(df[feat], errors="coerce").dropna().values.astype(float)

        if len(x) == 0:
            s_vec.append(0.0)
            e_vec.append(1.0)
            continue

        s = float(np.min(x))
        e = float(np.max(x))

        if np.isclose(s, e):
            delta = 1.0 if np.isclose(e, 0.0) else abs(e) * 0.1
            s -= delta
            e += delta

        s_vec.append(s)
        e_vec.append(e)

    return np.array(s_vec, dtype=float), np.array(e_vec, dtype=float)


def get_range(start, end, n_levels=5):
    vals = np.linspace(start, end, n_levels + 1)
    return vals[1:]


def shift_svec(s_vec, e_vec):
    return s_vec - (e_vec - s_vec) / 4


def putvec_in_range(result, s_vec, e_vec, n_levels=5):
    result = np.array(result, dtype=float)
    return n_levels * (result - s_vec) / (e_vec - s_vec)


def format_tick_values(values):
    out = []
    for v in values:
        if abs(v) >= 100:
            out.append(f"{v:.0f}")
        elif abs(v) >= 10:
            out.append(f"{v:.1f}")
        else:
            out.append(f"{v:.2f}")
    return out


def compute_summary_stats(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)

    return {
        "n": len(x),
        "mean": np.mean(x),
        "std": np.std(x, ddof=1) if len(x) > 1 else np.nan,
        "median": np.median(x),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": np.min(x),
        "max": np.max(x),
    }


def safe_shapiro(x: np.ndarray, max_n: int = 5000, random_seed: int = 2025):
    """
    Returns:
        W, p, n_used, note
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) < 3:
        return np.nan, np.nan, len(x), "too_few_samples"

    if np.allclose(x, x[0]):
        return np.nan, np.nan, len(x), "constant_values"

    if len(x) > max_n:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(x), size=max_n, replace=False)
        x_for_test = x[idx]
        note = f"subsampled_to_{max_n}"
    else:
        x_for_test = x
        note = "full_sample"

    W, p = stats.shapiro(x_for_test)
    return W, p, len(x_for_test), note


def shapiro_is_approximately_normal(p_value, note, alpha=0.05):
    """
    Operational rule:
    - the test must run successfully
    - p > alpha is treated as no significant deviation from normality
    """
    if pd.isna(p_value):
        return False
    if note in {"too_few_samples", "constant_values"}:
        return False
    return p_value > alpha


def safe_welch_ttest(x1: np.ndarray, x2: np.ndarray):
    """
    Returns:
        stat, p, note
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]

    if len(x1) < 2 or len(x2) < 2:
        return np.nan, np.nan, "too_few_samples"

    if np.allclose(x1, x1[0]) and np.allclose(x2, x2[0]):
        return np.nan, np.nan, "constant_values"

    try:
        stat, p = stats.ttest_ind(x1, x2, equal_var=False, nan_policy="omit")
        return stat, p, "welch_t_test"
    except Exception as e:
        return np.nan, np.nan, f"error:{type(e).__name__}"


def safe_mannwhitney(x1: np.ndarray, x2: np.ndarray):
    """
    Returns:
        stat, p, note
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]

    if len(x1) == 0 or len(x2) == 0:
        return np.nan, np.nan, "too_few_samples"

    try:
        stat, p = stats.mannwhitneyu(x1, x2, alternative="two-sided")
        return stat, p, "mannwhitney_u"
    except Exception as e:
        return np.nan, np.nan, f"error:{type(e).__name__}"


def choose_group_comparison_test(
    dep: np.ndarray,
    con: np.ndarray,
    dep_shapiro_p,
    dep_shapiro_note,
    con_shapiro_p,
    con_shapiro_note,
    alpha=0.05,
):
    """
    Decision rule:
    - if both groups are approximately normal -> Welch's t-test
    - otherwise -> two-sided Mann-Whitney U-test
    """
    dep_normal = shapiro_is_approximately_normal(dep_shapiro_p, dep_shapiro_note, alpha)
    con_normal = shapiro_is_approximately_normal(con_shapiro_p, con_shapiro_note, alpha)

    if dep_normal and con_normal:
        stat, p, note = safe_welch_ttest(dep, con)
        return {
            "test_selected": "welch_t_test",
            "test_statistic": stat,
            "test_p_value": p,
            "test_note": note,
            "dep_approx_normal": True,
            "con_approx_normal": True,
        }

    stat, p, note = safe_mannwhitney(dep, con)
    return {
        "test_selected": "mannwhitney_u",
        "test_statistic": stat,
        "test_p_value": p,
        "test_note": note,
        "dep_approx_normal": dep_normal,
        "con_approx_normal": con_normal,
    }


# =========================
# Radar plot
# =========================
class Radar:
    def __init__(self, figure, titles, labels, rect=None):
        if rect is None:
            rect = [0.06, 0.06, 0.88, 0.88]

        self.n = len(titles)
        self.angles = np.arange(0, 360, 360.0 / self.n)

        self.axes = [
            figure.add_axes(rect, projection="polar", label=f"axes{i}")
            for i in range(self.n)
        ]

        self.ax = self.axes[0]

        _, title_texts = self.ax.set_thetagrids(
            self.angles,
            labels=titles,
            fontsize=16
        )

        self.ax.tick_params(axis="x", pad=24)

        for txt in title_texts:
            txt.set_fontsize(16)
            txt.set_bbox(
                dict(
                    boxstyle="square,pad=0.28",
                    facecolor="#f2f2f2",
                    edgecolor="gray",
                    linewidth=1.2
                )
            )

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            _, radial_texts = ax.set_rgrids(
                range(1, 6),
                angle=angle,
                labels=label
            )
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(0, 5)

            for rtxt in radial_texts:
                rtxt.set_fontsize(8)
                rtxt.set_color("black")

    def plot(self, values, *args, **kwargs):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kwargs)


def plot_group_spider(
    df: pd.DataFrame,
    features,
    output_file: Path,
    title="Average NLP Feature Profiles by Group"
):
    if not features:
        print("[Warning] No valid features for spider plot.")
        return

    group_means = compute_group_means(df, features)
    s_vec, e_vec = compute_feature_ranges(df, features)
    s_vec_shifted = shift_svec(s_vec, e_vec)

    axis_labels = []
    for s, e in zip(s_vec_shifted, e_vec):
        ticks = get_range(s, e, n_levels=5)
        axis_labels.append(format_tick_values(ticks))

    spoke_titles = [FEATURE_LABELS.get(f, f) for f in features]

    fig = plt.figure(figsize=(13.5, 11.5))
    radar = Radar(fig, spoke_titles, axis_labels)

    style_map = {
        "normal": {"color": "#1f77b4", "label": GROUP_DISPLAY_NAMES["normal"]},
        "depressed": {"color": "#d62728", "label": GROUP_DISPLAY_NAMES["depressed"]},
    }

    for group in GROUP_ORDER:
        if group not in group_means.index:
            continue

        values = group_means.loc[group, features].values.astype(float)
        values_scaled = putvec_in_range(values, s_vec_shifted, e_vec, n_levels=5)

        radar.plot(
            values_scaled,
            "-",
            lw=2.8,
            color=style_map[group]["color"],
            alpha=0.9,
            label=style_map[group]["label"],
        )

    radar.ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.18, 1.14),
        frameon=True,
        fontsize=12
    )

    plt.title(title, y=1.12, fontsize=16)
    fig.savefig(output_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


# =========================
# Box plot
# =========================
def plot_group_boxplots(
    df: pd.DataFrame,
    features,
    output_file: Path,
    ncols=4,
    title="Box-plots of NLP Measures by Group"
):
    if not features:
        print("[Warning] No valid features for box plots.")
        return

    n_features = len(features)
    ncols = min(ncols, n_features)
    nrows = math.ceil(n_features / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, feat in enumerate(features):
        ax = axes[i]

        data = []
        labels = []
        for g in GROUP_ORDER:
            x = pd.to_numeric(
                df.loc[df["group"] == g, feat],
                errors="coerce"
            ).dropna().values
            data.append(x)
            labels.append(GROUP_DISPLAY_NAMES.get(g, g))

        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            widths=0.6,
            showfliers=False,
            medianprops=dict(color="red", linewidth=1.5),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            boxprops=dict(edgecolor="black", linewidth=1.0),
        )

        for patch in bp["boxes"]:
            patch.set_facecolor("#cfcfcf")

        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file, dpi=250)
    plt.close(fig)


# =========================
# Summary table
# =========================
def build_summary_table(df: pd.DataFrame, features):
    rows = []

    for feat in features:
        dep = pd.to_numeric(
            df.loc[df["group"] == "depressed", feat], errors="coerce"
        ).dropna().values.astype(float)

        con = pd.to_numeric(
            df.loc[df["group"] == "normal", feat], errors="coerce"
        ).dropna().values.astype(float)

        dep_stats = compute_summary_stats(dep)
        con_stats = compute_summary_stats(con)

        dep_W, dep_p, dep_n_used, dep_note = safe_shapiro(
            dep,
            max_n=SHAPIRO_MAX_N,
            random_seed=RANDOM_SEED,
        )
        con_W, con_p, con_n_used, con_note = safe_shapiro(
            con,
            max_n=SHAPIRO_MAX_N,
            random_seed=RANDOM_SEED,
        )

        comparison = choose_group_comparison_test(
            dep=dep,
            con=con,
            dep_shapiro_p=dep_p,
            dep_shapiro_note=dep_note,
            con_shapiro_p=con_p,
            con_shapiro_note=con_note,
            alpha=ALPHA_NORMALITY,
        )

        rows.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),

            "n_depressed": dep_stats["n"],
            "mean_depressed": dep_stats["mean"],
            "std_depressed": dep_stats["std"],
            "median_depressed": dep_stats["median"],
            "q1_depressed": dep_stats["q1"],
            "q3_depressed": dep_stats["q3"],
            "iqr_depressed": dep_stats["iqr"],
            "min_depressed": dep_stats["min"],
            "max_depressed": dep_stats["max"],

            "shapiro_W_depressed": dep_W,
            "shapiro_p_depressed": dep_p,
            "shapiro_n_used_depressed": dep_n_used,
            "shapiro_note_depressed": dep_note,
            "approx_normal_depressed": comparison["dep_approx_normal"],

            "n_normal": con_stats["n"],
            "mean_normal": con_stats["mean"],
            "std_normal": con_stats["std"],
            "median_normal": con_stats["median"],
            "q1_normal": con_stats["q1"],
            "q3_normal": con_stats["q3"],
            "iqr_normal": con_stats["iqr"],
            "min_normal": con_stats["min"],
            "max_normal": con_stats["max"],

            "shapiro_W_normal": con_W,
            "shapiro_p_normal": con_p,
            "shapiro_n_used_normal": con_n_used,
            "shapiro_note_normal": con_note,
            "approx_normal_normal": comparison["con_approx_normal"],

            "group_test_selected": comparison["test_selected"],
            "group_test_statistic": comparison["test_statistic"],
            "group_test_p_value": comparison["test_p_value"],
            "group_test_note": comparison["test_note"],
        })

    return pd.DataFrame(rows)


# =========================
# Main
# =========================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = build_dataframe(DEPRESSION_JSON, NORMAL_JSON)
    df = coerce_numeric(df, PLOT_FEATURES)

    feature_set_1 = resolve_plot_features(df, FEATURE_SET_1)
    feature_set_2 = resolve_plot_features(df, FEATURE_SET_2)

    plot_features = feature_set_1 + feature_set_2

    if not plot_features:
        print("No valid numeric features found. Please check your feature names.")
        return

    print("[Info] Feature set 1:")
    for f in feature_set_1:
        print("  -", f)

    print("[Info] Feature set 2:")
    for f in feature_set_2:
        print("  -", f)

    # 1) Spider plots
    if feature_set_1:
        plot_group_spider(
            df,
            feature_set_1,
            SPIDER_PNG_1,
            title="Average NLP Feature Profiles by Group (Part 1)"
        )

    if feature_set_2:
        plot_group_spider(
            df,
            feature_set_2,
            SPIDER_PNG_2,
            title="Average NLP Feature Profiles by Group (Part 2)"
        )

    # 2) Box plots
    if feature_set_1:
        plot_group_boxplots(df, feature_set_1, BOXPLOT_PNG_1, ncols=4)

    if feature_set_2:
        plot_group_boxplots(df, feature_set_2, BOXPLOT_PNG_2, ncols=4)

    # 3) Summary statistics + Shapiro-Wilk + adaptive group comparison
    summary_df = build_summary_table(df, plot_features)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("Done.")
    if feature_set_1:
        print(f"Spider plot 1 saved to: {SPIDER_PNG_1.resolve()}")
        print(f"Box plots 1 saved to: {BOXPLOT_PNG_1.resolve()}")
    if feature_set_2:
        print(f"Spider plot 2 saved to: {SPIDER_PNG_2.resolve()}")
        print(f"Box plots 2 saved to: {BOXPLOT_PNG_2.resolve()}")

    print(f"Summary table saved to: {SUMMARY_CSV.resolve()}")

    if not summary_df.empty:
        print("\n[Info] Selected group comparison tests:")
        print(summary_df["group_test_selected"].value_counts(dropna=False))


if __name__ == "__main__":
    main()