import os
import argparse
import numpy as np
import pandas as pd

from src.preprocessing.columns import *

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def check_missing_sofa_at_step0(states_df: pd.DataFrame) -> pd.DataFrame:
    if C_ICUSTAYID not in states_df.columns:
        raise ValueError(f"states_df missing column: {C_ICUSTAYID}")
    if C_TIMESTEP not in states_df.columns:
        raise ValueError(f"states_df missing column: {C_TIMESTEP}")
    if C_SOFA not in states_df.columns:
        raise ValueError(f"states_df missing column: {C_SOFA}")

    step0 = (
        states_df
        .sort_values([C_ICUSTAYID, C_TIMESTEP])
        .groupby(C_ICUSTAYID, as_index=False)
        .first()
    )

    n_total = len(step0)
    n_missing = int(step0[C_SOFA].isna().sum())

    print("\n=== CHECK SOFA AT FIRST STATE ===")
    print(f"Step-0 rows: {n_total}")
    print(f"Missing SOFA at step 0: {n_missing} ({100.0 * n_missing / max(n_total, 1):.2f}%)")

    return step0


def check_first_state_vs_first_sofa(states_df: pd.DataFrame, sofa_df: pd.DataFrame) -> pd.DataFrame:
    required_states = [C_ICUSTAYID, C_TIMESTEP]
    required_sofa = [C_ICUSTAYID, C_STARTTIME]

    for c in required_states:
        if c not in states_df.columns:
            raise ValueError(f"states_df missing column: {c}")

    for c in required_sofa:
        if c not in sofa_df.columns:
            raise ValueError(f"sofa_df missing column: {c}")

    first_state = (
        states_df
        .groupby(C_ICUSTAYID, as_index=False)[C_TIMESTEP]
        .min()
        .rename(columns={C_TIMESTEP: "first_state_time"})
    )

    first_sofa = (
        sofa_df
        .dropna(subset=[C_STARTTIME])
        .groupby(C_ICUSTAYID, as_index=False)[C_STARTTIME]
        .min()
        .rename(columns={C_STARTTIME: "first_sofa_starttime"})
    )

    merged = first_state.merge(first_sofa, on=C_ICUSTAYID, how="left")
    merged["has_sofa"] = merged["first_sofa_starttime"].notna()

    merged["relation"] = np.where(
        merged["first_sofa_starttime"].isna(),
        "no_sofa_for_stay",
        np.where(
            merged["first_state_time"] < merged["first_sofa_starttime"],
            "state_before_sofa",
            np.where(
                merged["first_state_time"] == merged["first_sofa_starttime"],
                "state_equals_sofa",
                "state_after_sofa",
            ),
        ),
    )

    merged["delta_hours_state_minus_sofa"] = (
        merged["first_state_time"] - merged["first_sofa_starttime"]
    ) / 3600.0

    counts = merged["relation"].value_counts(dropna=False).to_dict()

    print("\n=== CHECK FIRST STATE VS FIRST SOFA ===")
    print(f"Total stays: {len(merged)}")
    for k, v in counts.items():
        print(f"{k}: {v} ({100.0 * v / max(len(merged), 1):.2f}%)")

    subset = merged[merged["relation"] == "state_before_sofa"]
    if len(subset) > 0:
        print("\nDelta hours (state - sofa) for stays where state occurs before SOFA:")
        print(subset["delta_hours_state_minus_sofa"].describe())

    return merged


def print_example_stays(
    states_df: pd.DataFrame,
    sofa_df: pd.DataFrame,
    relation_df: pd.DataFrame,
    max_examples: int = 10,
):
    print("\n=== EXAMPLES OF PROBLEMATIC STAYS ===")

    bad = relation_df[
        relation_df["relation"].isin(["state_before_sofa", "no_sofa_for_stay"])
    ].copy()

    if len(bad) == 0:
        print("No problematic stays found.")
        return

    bad = bad.sort_values(
        by=["relation", "delta_hours_state_minus_sofa"],
        ascending=[True, True],
        na_position="last",
    ).head(max_examples)

    for _, row in bad.iterrows():
        stay_id = row[C_ICUSTAYID]
        rel = row["relation"]
        fst = row["first_state_time"]
        fsofa = row["first_sofa_starttime"]
        delta_h = row["delta_hours_state_minus_sofa"]

        print("\n----------------------------------------")
        print(f"stay_id: {stay_id}")
        print(f"relation: {rel}")
        print(f"first_state_time: {fst}")
        print(f"first_sofa_starttime: {fsofa}")
        print(f"delta_hours_state_minus_sofa: {delta_h}")

        stay_states = (
            states_df[states_df[C_ICUSTAYID] == stay_id]
            .sort_values(C_TIMESTEP)
            [[c for c in [C_BLOC, C_TIMESTEP, C_SOFA] if c in states_df.columns]]
            .head(10)
        )

        print("\nFirst states:")
        print(stay_states.to_string(index=False))

        stay_sofa_cols = [c for c in [C_ICUSTAYID, "hr", C_STARTTIME, C_ENDTIME, C_SOFA] if c in sofa_df.columns]
        stay_sofa = (
            sofa_df[sofa_df[C_ICUSTAYID] == stay_id]
            .sort_values(C_STARTTIME)
            [stay_sofa_cols]
            .head(10)
        )

        print("\nFirst SOFA intervals:")
        if len(stay_sofa) == 0:
            print("(no SOFA rows)")
        else:
            print(stay_sofa.to_string(index=False))


def inspect_single_stay(states_df: pd.DataFrame, sofa_df: pd.DataFrame, stay_id: int):
    print(f"\n=== DETAILED INSPECTION FOR STAY {stay_id} ===")

    stay_states = (
        states_df[states_df[C_ICUSTAYID] == stay_id]
        .sort_values(C_TIMESTEP)
        [[c for c in [C_BLOC, C_TIMESTEP, C_SOFA] if c in states_df.columns]]
    )

    stay_sofa_cols = [c for c in [C_ICUSTAYID, "hr", C_STARTTIME, C_ENDTIME, C_SOFA] if c in sofa_df.columns]
    stay_sofa = (
        sofa_df[sofa_df[C_ICUSTAYID] == stay_id]
        .sort_values(C_STARTTIME)
        [stay_sofa_cols]
    )

    print("\nSTATES:")
    if len(stay_states) == 0:
        print("(no rows)")
    else:
        print(stay_states.to_string(index=False))

    print("\nSOFA:")
    if len(stay_sofa) == 0:
        print("(no rows)")
    else:
        print(stay_sofa.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostics for alignment between patient_states and sofa_derived"
    )
    parser.add_argument(
        "--states",
        type=str,
        required=True,
        help="Path to patient_states.csv"
    )
    parser.add_argument(
        "--sofa",
        type=str,
        required=True,
        help="Path to sofa_derived.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="first_state_vs_first_sofa_check.csv",
        help="Output CSV with stay-level comparison"
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Number of example stays to print"
    )
    parser.add_argument(
        "--stay-id",
        type=int,
        default=None,
        help="If specified, prints full details for a single stay"
    )
    args = parser.parse_args()

    print("Reading files...")
    states_df = pd.read_csv(args.states)
    sofa_df = pd.read_csv(args.sofa)

    step0_df = check_missing_sofa_at_step0(states_df)
    relation_df = check_first_state_vs_first_sofa(states_df, sofa_df)

    merged = relation_df.merge(
        step0_df[[C_ICUSTAYID, C_SOFA]].rename(columns={C_SOFA: "step0_sofa"}),
        on=C_ICUSTAYID,
        how="left"
    )
    merged["step0_sofa_missing"] = merged["step0_sofa"].isna()

    print("\n=== CROSS TAB RELATION / MISSING SOFA AT STEP 0 ===")
    cross = pd.crosstab(merged["relation"], merged["step0_sofa_missing"])
    print(cross)

    print_example_stays(
        states_df=states_df,
        sofa_df=sofa_df,
        relation_df=merged,
        max_examples=args.examples,
    )

    if args.stay_id is not None:
        inspect_single_stay(states_df, sofa_df, args.stay_id)

    merged.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()