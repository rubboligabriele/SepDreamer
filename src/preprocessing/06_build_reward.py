import os
import argparse

from preprocessing.utils.columns import *
from preprocessing.utils.utils import load_csv, fit_action_bins, transform_actions_separate, save_pickle
from preprocessing.reward.reward_medR import add_medr_reward_to_dataframe

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

DEFAULT_REWARD_COL = "reward"

IV_BIN_COL = "iv_fluid_5quantile"
VASO_BIN_COL = "vaso_5quantile"


def merge_outcome_if_needed(df, outcome_df=None, outcome_col=C_MORTA_90):
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
        raise ValueError(f"Outcome file is missing required columns: {missing_cols}")

    outcome_one_row = (
        outcome_df[[C_ICUSTAYID, outcome_col]]
        .drop_duplicates(subset=C_ICUSTAYID)
    )

    return df.merge(outcome_one_row, how="left", on=C_ICUSTAYID, suffixes=("", "_outcome"))


def add_binned_action_columns(
    actions_df,
    fluid_col=C_INPUT_STEP,
    vaso_col=C_MAX_DOSE_VASO,
    num_action_bins=5,
):
    actions_df = actions_df.copy()

    _, action_medians, action_cutoffs = fit_action_bins(
        actions_df[fluid_col].astype(float).fillna(0.0).values,
        actions_df[vaso_col].astype(float).fillna(0.0).values,
        n_action_bins=num_action_bins,
    )

    iv_bins, vaso_bins = transform_actions_separate(
        actions_df[fluid_col].astype(float).fillna(0.0).values,
        actions_df[vaso_col].astype(float).fillna(0.0).values,
        action_cutoffs,
    )

    actions_df["iv_fluid_5quantile"] = iv_bins
    actions_df["vaso_5quantile"] = vaso_bins

    return actions_df, action_medians, action_cutoffs


def main():
    parser = argparse.ArgumentParser(
        description="Add medR reward column to a patient-states dataframe."
    )

    parser.add_argument("input", type=str, help="Input states CSV, e.g. patient_states_filtered.csv")
    parser.add_argument("output", type=str, help="Output states CSV with reward")

    parser.add_argument("--actions", type=str, required=True, help="Path to actions_filtered.csv")
    parser.add_argument("--delta-fresh", type=str, required=True, help="Path to delta_fresh_filtered.csv")

    parser.add_argument("--outcome-file", type=str, default=None)
    parser.add_argument("--outcome-col", type=str, default=C_MORTA_90)

    parser.add_argument("--fluid-col", type=str, default=C_INPUT_STEP)
    parser.add_argument("--vaso-col", type=str, default=C_MAX_DOSE_VASO)
    parser.add_argument("--num-action-bins", type=int, default=5)

    parser.add_argument("--reward-col", type=str, default=DEFAULT_REWARD_COL)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument(
        "--action-bin-config-out",
        type=str,
        default=None,
        help="Optional path to save medR action bin config used for reward."
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Reading states...")
    states_df = load_csv(args.input)

    print("Reading actions...")
    actions_df = load_csv(args.actions)

    print("Reading freshness delta...")
    delta_fresh_df = load_csv(args.delta_fresh)

    outcome_df = None
    if args.outcome_file is not None:
        print("Reading outcome dataframe...")
        outcome_df = load_csv(args.outcome_file)

    print("Checking outcome column...")
    states_df = merge_outcome_if_needed(
        states_df,
        outcome_df=outcome_df,
        outcome_col=args.outcome_col,
    )

    print("Binning actions for medR reward...")
    actions_df, action_medians, action_cutoffs = add_binned_action_columns(
        actions_df,
        fluid_col=args.fluid_col,
        vaso_col=args.vaso_col,
        num_action_bins=args.num_action_bins,
    )

    if args.action_bin_config_out is not None:
        os.makedirs(os.path.dirname(args.action_bin_config_out), exist_ok=True)
        save_pickle(
            args.action_bin_config_out,
            {
                "num_action_bins": args.num_action_bins,
                "fluid_col": args.fluid_col,
                "vaso_col": args.vaso_col,
                "action_cutoffs": action_cutoffs,
                "action_medians": action_medians,
                "iv_bin_col": IV_BIN_COL,
                "vaso_bin_col": VASO_BIN_COL,
            },
        )
        print(f"Saved action bin config to {args.action_bin_config_out}")

    print("Computing medR rewards...")
    states_df = add_medr_reward_to_dataframe(
        states_df=states_df,
        actions_df=actions_df,
        delta_fresh_df=delta_fresh_df,
        reward_col=args.reward_col,
        gamma=args.gamma,
    )

    n_nan = int(states_df[args.reward_col].isna().sum())
    if n_nan > 0:
        print(f"WARNING: found {n_nan} NaN rewards")
    else:
        print("Reward computation finished. No NaN rewards.")

    print("Writing output...")
    states_df.to_csv(args.output, index=False)

    print("Done.")


if __name__ == "__main__":
    main()