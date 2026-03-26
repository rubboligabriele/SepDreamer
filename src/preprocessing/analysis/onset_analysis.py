import os
import argparse
import numpy as np
import pandas as pd

from src.preprocessing.columns import *
from src.preprocessing.utils import load_csv


# =========================================================
# AI CLINICIAN ONSET LOGIC
# =========================================================

def calculate_onset(abx, bacterio, stay_id):

    matching_abs = (
        abx.loc[abx[C_ICUSTAYID] == stay_id, C_STARTDATE]
        .dropna()
        .sort_values()
    )

    matching_bacts = (
        bacterio[bacterio[C_ICUSTAYID] == stay_id]
        .dropna(subset=[C_CHARTTIME])
        .sort_values(C_CHARTTIME)
    )

    if matching_abs.empty or matching_bacts.empty:
        return None

    for ab_time in matching_abs:

        dists = np.abs(matching_bacts[C_CHARTTIME].values - ab_time) / 3600
        min_idx = np.argmin(dists)

        bact_time = matching_bacts.iloc[min_idx][C_CHARTTIME]

        # antibiotic first
        if dists[min_idx] <= 24 and ab_time <= bact_time:
            return {
                C_ICUSTAYID: stay_id,
                "onset_ai": ab_time
            }

        # culture first
        if dists[min_idx] <= 72 and ab_time >= bact_time:
            return {
                C_ICUSTAYID: stay_id,
                "onset_ai": bact_time
            }

    return None


# =========================================================
# SOFA FILTER
# =========================================================

def has_valid_sofa_block(sofa, stay_id, onset_time):

    sofa_stay = sofa[
        (sofa[C_ICUSTAYID] == stay_id) &
        (sofa[C_SOFA] >= 2)
    ]

    if sofa_stay.empty:
        return False

    lower = onset_time - 48 * 3600
    upper = onset_time + 24 * 3600

    valid = sofa_stay[
        (sofa_stay[C_ENDTIME] >= lower) &
        (sofa_stay[C_ENDTIME] <= upper)
    ]

    return len(valid) > 0


def calculate_onset_ai_with_sofa(abx, bacterio, sofa, stay_id):

    onset = calculate_onset(abx, bacterio, stay_id)

    if onset is None:
        return None

    onset_time = onset["onset_ai"]

    if not has_valid_sofa_block(sofa, stay_id, onset_time):
        return None

    return onset

def analyze_delta_distribution(cmp_df, output_dir, suffix):

    both = cmp_df[cmp_df["group"] == "both"].copy()

    if both.empty:
        print("No overlapping onsets → skip delta analysis")
        return

    delta = both["delta_h"].dropna()
    abs_delta = both["abs_delta_h"].dropna()

    stats = {
        "n": len(delta),
        "mean_delta": delta.mean(),
        "median_delta": delta.median(),
        "p5": delta.quantile(0.05),
        "p10": delta.quantile(0.10),
        "p25": delta.quantile(0.25),
        "p75": delta.quantile(0.75),
        "p90": delta.quantile(0.90),
        "p95": delta.quantile(0.95),

        "mean_abs_delta": abs_delta.mean(),
        "median_abs_delta": abs_delta.median(),
        "p90_abs_delta": abs_delta.quantile(0.90),
        "p95_abs_delta": abs_delta.quantile(0.95),
    }

    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(
        os.path.join(output_dir, f"onset_delta_distribution_summary_{suffix}.csv"),
        index=False
    )

    bins = [-1e9, -48, -24, -12, -6, -3, -1, 1, 3, 6, 12, 24, 48, 1e9]
    labels = [
        "<-48","-48:-24","-24:-12","-12:-6","-6:-3",
        "-3:-1","-1:1","1:3","3:6","6:12","12:24","24:48",">48"
    ]

    both["delta_bin"] = pd.cut(delta, bins=bins, labels=labels)

    hist = both["delta_bin"].value_counts().sort_index().reset_index()
    hist.columns = ["bin", "count"]
    hist["pct"] = hist["count"] / hist["count"].sum() * 100

    hist.to_csv(
        os.path.join(output_dir, f"onset_delta_histogram_{suffix}.csv"),
        index=False
    )

    print(f"Saved delta distribution analysis ({suffix})")


# =========================================================
# BUILD TABLES
# =========================================================

def build_ai_onset(abx, bacterio):

    stay_ids = sorted(
        set(abx[C_ICUSTAYID].unique()) &
        set(bacterio[C_ICUSTAYID].unique())
    )

    rows = []

    for stay in stay_ids:
        o = calculate_onset(abx, bacterio, stay)
        if o:
            rows.append(o)

    return pd.DataFrame(rows).drop_duplicates(subset=[C_ICUSTAYID])


def build_ai_onset_with_sofa(abx, bacterio, sofa):

    stay_ids = sorted(
        set(abx[C_ICUSTAYID].unique()) &
        set(bacterio[C_ICUSTAYID].unique())
    )

    rows = []

    for stay in stay_ids:
        o = calculate_onset_ai_with_sofa(abx, bacterio, sofa, stay)
        if o:
            rows.append(o)

    return pd.DataFrame(rows).drop_duplicates(subset=[C_ICUSTAYID])


# =========================================================
# DERIVED LOAD
# =========================================================

def load_derived_onset(path):

    df = load_csv(path)

    if C_ONSET_TIME in df.columns:
        df = df.rename(columns={C_ONSET_TIME: "onset_derived"})
    elif "suspected_infection_time" in df.columns:
        df = df.rename(columns={"suspected_infection_time": "onset_derived"})
    else:
        raise ValueError("Derived onset column not found")

    return df[[C_ICUSTAYID, "onset_derived"]].drop_duplicates()


# =========================================================
# COMPARISON
# =========================================================

def compare(ai, derived):

    df = pd.merge(ai, derived, on=C_ICUSTAYID, how="outer")

    df["has_ai"] = ~df["onset_ai"].isna()
    df["has_derived"] = ~df["onset_derived"].isna()

    df["delta_h"] = (df["onset_derived"] - df["onset_ai"]) / 3600
    df["abs_delta_h"] = df["delta_h"].abs()

    df["group"] = np.select(
        [
            df["has_ai"] & df["has_derived"],
            df["has_ai"] & ~df["has_derived"],
            ~df["has_ai"] & df["has_derived"]
        ],
        [
            "both",
            "only_ai",
            "only_derived"
        ],
        default="none"
    )

    return df


def summarize(df):

    both = df[df["group"] == "both"]

    return pd.DataFrame([{
        "n_total": len(df),
        "n_both": len(both),
        "n_only_ai": (df["group"] == "only_ai").sum(),
        "n_only_derived": (df["group"] == "only_derived").sum(),
        "median_abs_delta_h": both["abs_delta_h"].median(),
        "p90_abs_delta_h": both["abs_delta_h"].quantile(0.9),
        "same_24h_pct": (both["abs_delta_h"] <= 24).mean() * 100
    }])


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--abx", required=True)
    parser.add_argument("--bacterio", required=True)
    parser.add_argument("--sofa", required=True)
    parser.add_argument("--derived", required=True)
    parser.add_argument("--outdir", required=True)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading data")

    abx = load_csv(args.abx).dropna(subset=[C_ICUSTAYID, C_STARTDATE])
    bacterio = load_csv(args.bacterio).dropna(subset=[C_ICUSTAYID, C_CHARTTIME])
    sofa = load_csv(args.sofa).dropna(subset=[C_ICUSTAYID, C_ENDTIME, C_SOFA])

    derived = load_derived_onset(args.derived)

    print("Building AI onset")
    ai = build_ai_onset(abx, bacterio)

    print("Building AI+SOFA onset")
    ai_sofa = build_ai_onset_with_sofa(abx, bacterio, sofa)

    print("Comparing")

    cmp_ai = compare(ai, derived)
    cmp_ai_sofa = compare(ai_sofa, derived)

    sum_ai = summarize(cmp_ai)
    sum_ai_sofa = summarize(cmp_ai_sofa)

    analyze_delta_distribution(cmp_ai, args.outdir, "onset_ai")
    analyze_delta_distribution(cmp_ai_sofa, args.outdir, "onset_ai_sofa")

    ai.to_csv(os.path.join(args.outdir, "onset_ai.csv"), index=False)
    ai_sofa.to_csv(os.path.join(args.outdir, "onset_ai_sofa.csv"), index=False)

    cmp_ai.to_csv(os.path.join(args.outdir, "comparison_ai.csv"), index=False)
    cmp_ai_sofa.to_csv(os.path.join(args.outdir, "comparison_ai_sofa.csv"), index=False)

    sum_ai.to_csv(os.path.join(args.outdir, "summary_ai.csv"), index=False)
    sum_ai_sofa.to_csv(os.path.join(args.outdir, "summary_ai_sofa.csv"), index=False)

    print("\nDONE")