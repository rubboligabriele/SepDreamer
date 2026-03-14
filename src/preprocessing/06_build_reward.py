import os
import argparse
import numpy as np
import pandas as pd

from src.preprocessing.columns import *
from src.preprocessing.utils import load_csv
from src.preprocessing.reward import add_reward_to_dataframe

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

DEFAULT_REWARD_COL = "reward"


def merge_outcome_if_needed(
    df,
    outcome_df=None,
    outcome_col=C_MORTA_90,
):
    """
    Ensure the main dataframe contains the outcome column needed for terminal reward.

    If the outcome column is already present in df, do nothing.
    Otherwise, merge it from outcome_df using icustayid.
    """
    if outcome_col in df.columns:
        return df

    if outcome_df is None:
        raise ValueError(
            f"Column '{outcome_col}' is not present in the input dataframe, "
            "and no outcome file was provided."
        )

    required_cols = [C_ICUSTAYID, outcome_col]
    missing_cols = [c for c in required_cols if c not in outcome_df.columns]
    if missing_cols:
        raise ValueError(
            f"Outcome file is missing required columns: {missing_cols}"
        )

    outcome_one_row = outcome_df[[C_ICUSTAYID, outcome_col]].drop_duplicates(subset=C_ICUSTAYID)

    df = df.merge(
        outcome_one_row,
        how="left",
        on=C_ICUSTAYID,
        suffixes=("", "_outcome")
    )

    return df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add sepsis reward column to a patient-states dataframe. "
            "Intermediate rewards follow the SOFA/lactate formulation, "
            "terminal rewards use the outcome column."
        )
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input CSV path (typically patient_states_filtered.csv)"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output CSV path"
    )

    parser.add_argument(
        "--outcome-file",
        dest="outcome_file",
        type=str,
        default=None,
        help=(
            "Optional CSV containing icustayid and outcome column "
            f"(e.g. sepsis_cohort.csv with {C_MORTA_90}). "
            "Used only if outcome is not already present in input."
        )
    )

    parser.add_argument(
        "--outcome-col",
        dest="outcome_col",
        type=str,
        default=C_MORTA_90,
        help=f"Outcome column name for terminal reward (default: {C_MORTA_90})"
    )

    parser.add_argument(
        "--sofa-col",
        dest="sofa_col",
        type=str,
        default=C_SOFA,
        help=f"SOFA column name (default: {C_SOFA})"
    )

    parser.add_argument(
        "--lactate-col",
        dest="lactate_col",
        type=str,
        default=C_ARTERIAL_LACTATE,
        help=f"Lactate column name (default: {C_ARTERIAL_LACTATE})"
    )

    parser.add_argument(
        "--reward-col",
        dest="reward_col",
        type=str,
        default=DEFAULT_REWARD_COL,
        help=f"Reward column name to write (default: {DEFAULT_REWARD_COL})"
    )

    parser.add_argument(
        "--c0",
        dest="c0",
        type=float,
        default=-0.025,
        help="SOFA stagnation penalty coefficient"
    )
    parser.add_argument(
        "--c1",
        dest="c1",
        type=float,
        default=-0.125,
        help="SOFA delta coefficient"
    )
    parser.add_argument(
        "--c2",
        dest="c2",
        type=float,
        default=-2.0,
        help="Lactate tanh delta coefficient"
    )
    parser.add_argument(
        "--r-terminal",
        dest="r_terminal",
        type=float,
        default=15.0,
        help="Absolute terminal reward"
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Reading input dataframe...")
    df = load_csv(args.input)

    outcome_df = None
    if args.outcome_file is not None:
        print("Reading outcome dataframe...")
        outcome_df = load_csv(args.outcome_file)

    print("Checking outcome column...")
    df = merge_outcome_if_needed(
        df,
        outcome_df=outcome_df,
        outcome_col=args.outcome_col,
    )

    print("Computing rewards...")
    df = add_reward_to_dataframe(
        df,
        outcome_col=args.outcome_col,
        sofa_col=args.sofa_col,
        lactate_col=args.lactate_col,
        reward_col=args.reward_col,
        c0=args.c0,
        c1=args.c1,
        c2=args.c2,
        r_terminal=args.r_terminal,
        missing_strategy=None,   # leave NaN for now if required values are missing
    )

    n_nan = int(df[args.reward_col].isna().sum())
    print(f"Reward computation finished. NaN rewards: {n_nan}")

    print("Writing output...")
    df.to_csv(args.output, index=False, float_format="%g")

    print("Done.")


if __name__ == "__main__":
    main()