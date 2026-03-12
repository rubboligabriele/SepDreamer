import os
import argparse
import pandas as pd

from src.preprocessing.columns import *
from src.preprocessing.utils import load_csv

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def outlier_stay_ids(actions):
    outliers = set()

    if C_OUTPUT_STEP in actions.columns:
        outliers |= set(actions.loc[actions[C_OUTPUT_STEP] > 12000, C_ICUSTAYID].unique())

    if C_INPUT_STEP in actions.columns:
        outliers |= set(actions.loc[actions[C_INPUT_STEP] > 10000, C_ICUSTAYID].unique())

    return outliers


def build_sepsis_cohort(states, demog):
    """
    Build one-row-per-stay cohort table with outcome and severity.
    """

    # summarise temporal states
    state_summary = states.groupby(C_ICUSTAYID).agg({
        C_SOFA: "max",
        C_SIRS: "max",
    }).rename({
        C_SOFA: C_MAX_SOFA,
        C_SIRS: C_MAX_SIRS,
    }, axis=1)

    # outcome from demog
    outcome = demog[[C_ICUSTAYID, C_MORTA_90]].drop_duplicates(subset=C_ICUSTAYID)
    outcome = outcome.set_index(C_ICUSTAYID)

    # merge
    sepsis = state_summary.join(outcome, how="inner")

    # sepsis definition
    sepsis = sepsis[sepsis[C_MAX_SOFA] >= 2]

    return sepsis


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Build MedDreamer sepsis cohort and filter states/actions/mask/delta."
    )

    parser.add_argument("--states", required=True, type=str)
    parser.add_argument("--actions", required=True, type=str)
    parser.add_argument("--demog", required=True, type=str)

    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--delta", type=str, default=None)
    parser.add_argument("--qstime", type=str, default=None)

    parser.add_argument("--output-dir", required=True, type=str)

    parser.add_argument(
        "--no-outlier-exclusion",
        dest="outlier_exclusion",
        action="store_false",
        default=True
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Reading files...")

    states = load_csv(args.states)
    actions = load_csv(args.actions)
    demog = load_csv(args.demog)

    mask = load_csv(args.mask) if args.mask else None
    delta = load_csv(args.delta) if args.delta else None
    qstime = load_csv(args.qstime) if args.qstime else None

    print("Building sepsis cohort from patient_states + demog...")

    sepsis = build_sepsis_cohort(states, demog)

    print(f"SOFA >= 2 stays: {len(sepsis)}")

    if args.outlier_exclusion:
        outliers = outlier_stay_ids(actions)
        print(f"Outlier stays to remove: {len(outliers)}")
        sepsis = sepsis[~sepsis.index.isin(outliers)]

    if qstime is not None:
        qstime = qstime.set_index(C_ICUSTAYID, drop=True)

        if C_ONSET_TIME in qstime.columns:
            sepsis = pd.merge(
                sepsis,
                qstime[[C_ONSET_TIME]],
                how="left",
                left_index=True,
                right_index=True
            )

    keep_ids = set(sepsis.index)

    print(f"Final cohort stays: {len(keep_ids)}")

    states_f = states[states[C_ICUSTAYID].isin(keep_ids)].copy()
    actions_f = actions[actions[C_ICUSTAYID].isin(keep_ids)].copy()

    states_f.to_csv(
        os.path.join(args.output_dir, "patient_states_filtered.csv"),
        index=False
    )

    actions_f.to_csv(
        os.path.join(args.output_dir, "actions_filtered.csv"),
        index=False
    )

    sepsis.to_csv(
        os.path.join(args.output_dir, "sepsis_cohort.csv")
    )

    if mask is not None:
        mask_f = mask[mask[C_ICUSTAYID].isin(keep_ids)].copy()
        mask_f.to_csv(
            os.path.join(args.output_dir, "mask_filtered.csv"),
            index=False
        )

    if delta is not None:
        delta_f = delta[delta[C_ICUSTAYID].isin(keep_ids)].copy()
        delta_f.to_csv(
            os.path.join(args.output_dir, "delta_filtered.csv"),
            index=False
        )

    print("Done.")