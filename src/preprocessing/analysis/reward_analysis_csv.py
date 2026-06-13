import os
import argparse
import numpy as np
import pandas as pd

from preprocessing.utils.columns import *
from preprocessing.utils.utils import load_csv

DEFAULT_REWARD_COL = "reward"

MEDR_FEATURES = {
    "sofa_24hours": C_SOFA,
    "baseexcess": C_ARTERIAL_BE,
    "lactate": C_ARTERIAL_LACTATE,
    "urineoutput": C_URINE_OUTPUT,
    "mbp": C_MEANBP,
    "heartrate": C_HR,
}

IV_BIN_COL = "iv_fluid_5quantile"
VASO_BIN_COL = "vaso_5quantile"


def print_stats(name, x):
    x = np.asarray(x, dtype=float)
    print(f"\n{name}")
    print(f"  count: {np.isfinite(x).sum()}")
    print(f"  NaN:   {np.isnan(x).sum()}")

    if np.isfinite(x).sum() == 0:
        return

    print(f"  min:   {np.nanmin(x):.6f}")
    print(f"  max:   {np.nanmax(x):.6f}")
    print(f"  mean:  {np.nanmean(x):.6f}")
    print(f"  std:   {np.nanstd(x):.6f}")

    for q in [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]:
        print(f"  q={q:>4}: {np.nanquantile(x, q):.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze medR reward distribution and related signals."
    )

    parser.add_argument("states", type=str, help="Path to patient_states_with_reward.csv")
    parser.add_argument("--actions", type=str, default=None, help="Optional actions CSV with binned medR actions")
    parser.add_argument("--delta-fresh", type=str, default=None, help="Optional delta_fresh CSV")
    parser.add_argument("--reward-col", type=str, default=DEFAULT_REWARD_COL)

    args = parser.parse_args()

    print("Reading states...")
    df = load_csv(args.states)
    df = df.sort_values([C_ICUSTAYID, C_TIMESTEP]).reset_index(drop=True)

    if args.reward_col not in df.columns:
        raise ValueError(f"Missing reward column: {args.reward_col}")

    rewards = df[args.reward_col].to_numpy(dtype=float)

    print("\n=== GLOBAL medR REWARD STATS ===")
    print(f"Rows: {len(df)}")
    print(f"ICU stays: {df[C_ICUSTAYID].nunique()}")
    print_stats("Reward", rewards)

    last_idx = df.groupby(C_ICUSTAYID, sort=False).tail(1).index
    first_idx = df.groupby(C_ICUSTAYID, sort=False).head(1).index

    first_mask = df.index.isin(first_idx)
    last_mask = df.index.isin(last_idx)
    middle_mask = ~(first_mask | last_mask)

    print("\n=== POSITION IN EPISODE ===")
    print(f"First rows:  {first_mask.sum()}")
    print(f"Middle rows: {middle_mask.sum()}")
    print(f"Last rows:   {last_mask.sum()}")

    print_stats("First-row rewards", df.loc[first_mask, args.reward_col].values)
    print_stats("Middle-row rewards", df.loc[middle_mask, args.reward_col].values)
    print_stats("Last-row rewards", df.loc[last_mask, args.reward_col].values)

    print("\n=== REWARD SIGNS ===")
    print(f"reward == 0: {np.isclose(rewards, 0.0, atol=1e-8).sum()}")
    print(f"reward > 0:  {(rewards > 0).sum()}")
    print(f"reward < 0:  {(rewards < 0).sum()}")

    print("\n=== TOP 20 MOST COMMON REWARDS ===")
    vc = pd.Series(np.round(rewards, 6)).value_counts().head(20)
    print(vc.to_string())

    print("\n=== medR FEATURE VALUE STATS ===")
    for name, col in MEDR_FEATURES.items():
        if col not in df.columns:
            print(f"\n{name}: missing column {col}")
            continue
        print_stats(f"{name} ({col})", df[col].dropna().values)

    if args.delta_fresh is not None:
        print("\nReading delta freshness...")
        delta = load_csv(args.delta_fresh)
        delta = delta.sort_values([C_ICUSTAYID, C_TIMESTEP]).reset_index(drop=True)

        print("\n=== medR FRESHNESS DELTA STATS ===")
        for name, col in MEDR_FEATURES.items():
            if col not in delta.columns:
                print(f"\n{name}: missing delta column {col}")
                continue
            print_stats(f"freshness {name} ({col})", delta[col].dropna().values)

    if args.actions is not None:
        print("\nReading actions...")
        actions = load_csv(args.actions)
        actions = actions.sort_values([C_ICUSTAYID, C_TIMESTEP]).reset_index(drop=True)

        print("\n=== ACTION STATS ===")

        if IV_BIN_COL in actions.columns:
            print("\niv_fluid_5quantile counts:")
            print(actions[IV_BIN_COL].value_counts(dropna=False).sort_index().to_string())
        else:
            print(f"\nMissing column: {IV_BIN_COL}")

        if VASO_BIN_COL in actions.columns:
            print("\nvaso_5quantile counts:")
            print(actions[VASO_BIN_COL].value_counts(dropna=False).sort_index().to_string())
        else:
            print(f"\nMissing column: {VASO_BIN_COL}")

        if IV_BIN_COL in actions.columns and VASO_BIN_COL in actions.columns:
            print("\nJoint action bin counts:")
            joint = (
                actions.groupby([IV_BIN_COL, VASO_BIN_COL])
                .size()
                .reset_index(name="count")
                .sort_values([IV_BIN_COL, VASO_BIN_COL])
            )
            print(joint.to_string(index=False))

            penalty = (
                (actions[IV_BIN_COL].astype(float) / 4.0) * 0.25 +
                (actions[VASO_BIN_COL].astype(float) / 4.0) * 0.25
            )
            print_stats("Estimated medR action penalty", penalty.values)

    print("\nDone.")


if __name__ == "__main__":
    main()