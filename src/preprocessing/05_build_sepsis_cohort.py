import os
import argparse
import numpy as np
import pandas as pd

from src.preprocessing.columns import *
from src.preprocessing.utils import load_csv

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def outlier_stay_ids(states, actions):
    outliers = set()

    if C_OUTPUT_STEP in actions.columns:
        outliers |= set(
            actions.loc[actions[C_OUTPUT_STEP] > 12000, C_ICUSTAYID].dropna().unique()
        )

    if C_INPUT_STEP in actions.columns:
        outliers |= set(
            actions.loc[actions[C_INPUT_STEP] > 10000, C_ICUSTAYID].dropna().unique()
        )

    if C_TOTAL_BILI in states.columns:
        outliers |= set(
            states.loc[states[C_TOTAL_BILI] > 10000, C_ICUSTAYID].dropna().unique()
        )

    return outliers


def treatment_stopped_stay_ids(states, demog, actions, max_short_coverage_hours=80):
    """
    Adaptation of AI Clinician withdrawal / treatment stopped exclusion.

    Remove stays where:
    - morta_90 == 1
    - last vaso is NaN or < 0.01
    - max vaso during stay > 0.3
    - last SOFA >= half of max SOFA
    - coverage_hours < max_short_coverage_hours
    """
    needed_state_cols = [C_BLOC, C_ICUSTAYID, C_SOFA, C_TIMESTEP]
    needed_action_cols = [C_BLOC, C_ICUSTAYID, C_MAX_DOSE_VASO]
    needed_demog_cols = [C_ICUSTAYID, C_MORTA_90]

    for col in needed_state_cols:
        if col not in states.columns:
            raise ValueError(f"Missing column in states: {col}")

    for col in needed_action_cols:
        if col not in actions.columns:
            raise ValueError(f"Missing column in actions: {col}")

    for col in needed_demog_cols:
        if col not in demog.columns:
            raise ValueError(f"Missing column in demog: {col}")

    demog_small = demog[[C_ICUSTAYID, C_MORTA_90]].drop_duplicates(subset=[C_ICUSTAYID])

    a = pd.merge(
        states[[C_BLOC, C_ICUSTAYID, C_SOFA]],
        actions[[C_BLOC, C_ICUSTAYID, C_MAX_DOSE_VASO]],
        on=[C_ICUSTAYID, C_BLOC],
        how="inner"
    )

    a = pd.merge(
        a,
        demog_small,
        on=C_ICUSTAYID,
        how="left"
    )

    if len(a) == 0:
        return set()

    grouped = a.groupby(C_ICUSTAYID)

    d = grouped.agg({
        C_SOFA: "max",
        C_MAX_DOSE_VASO: "max",
        C_MORTA_90: "max",
    })

    last_bloc = (
        a.sort_values([C_ICUSTAYID, C_BLOC], ascending=[True, False])
         .drop_duplicates(C_ICUSTAYID)
         .rename(columns={
             C_MAX_DOSE_VASO: C_LAST_VASO,
             C_SOFA: C_LAST_SOFA
         })
    )[[C_ICUSTAYID, C_LAST_VASO, C_LAST_SOFA]]

    coverage = (
        states.groupby(C_ICUSTAYID)[C_TIMESTEP]
        .agg(["min", "max"])
        .rename(columns={"min": "first_timestep", "max": "last_timestep"})
    )
    coverage["coverage_hours"] = (
        coverage["last_timestep"] - coverage["first_timestep"]
    ) / 3600.0

    d = pd.merge(
        d,
        last_bloc,
        how="left",
        left_index=True,
        right_on=C_ICUSTAYID
    ).set_index(C_ICUSTAYID, drop=True)

    d = d.merge(coverage[["coverage_hours"]], left_index=True, right_index=True, how="left")

    stopped_treatment = d[
        (d[C_MORTA_90] == 1) &
        (pd.isna(d[C_LAST_VASO]) | (d[C_LAST_VASO] < 0.01)) &
        (d[C_MAX_DOSE_VASO] > 0.3) &
        (d[C_LAST_SOFA] >= d[C_SOFA] / 2) &
        (d["coverage_hours"] < max_short_coverage_hours)
    ].index

    return set(stopped_treatment)


def missing_static_stay_ids(states):
    """
    Remove a stay only if at least one static feature is missing for the whole stay.
    """
    static_cols = [
        C_AGE,
        C_GENDER,
        C_ELIXHAUSER,
        C_WEIGHT,
        C_RE_ADMISSION,
    ]

    bad = set()

    for stay, g in states.groupby(C_ICUSTAYID):
        for col in static_cols:
            if col not in g.columns:
                continue

            if g[col].notna().sum() == 0:
                bad.add(stay)
                break

    return bad


def missing_outcome_stay_ids(demog):
    if C_MORTA_90 not in demog.columns:
        raise ValueError(f"Missing mortality column: {C_MORTA_90}")

    return set(
        demog.loc[demog[C_MORTA_90].isna(), C_ICUSTAYID].dropna().unique()
    )


def died_in_icu_during_collection_stay_ids(demog, qstime):
    """
    AI Clinician-like exclusion:
    remove patients who die in ICU / very close to ICU outtime
    while still inside the data collection period.
    """
    needed_demog = [C_ICUSTAYID, C_DOD, C_OUTTIME, C_DISCHTIME]
    needed_qstime = [C_ICUSTAYID, C_LAST_TIMESTEP]

    for col in needed_demog:
        if col not in demog.columns:
            raise ValueError(f"Missing column in demog: {col}")

    for col in needed_qstime:
        if col not in qstime.columns:
            raise ValueError(f"Missing column in qstime: {col}")

    d = demog[needed_demog].drop_duplicates(subset=[C_ICUSTAYID]).copy()
    q = qstime[[C_ICUSTAYID, C_LAST_TIMESTEP]].drop_duplicates(subset=[C_ICUSTAYID]).copy()

    x = pd.merge(d, q, on=C_ICUSTAYID, how="inner")

    x[C_DIED_WITHIN_48H_OF_OUT_TIME] = (
        x[C_DOD].notna() &
        x[C_OUTTIME].notna() &
        ((x[C_DOD] - x[C_OUTTIME]).abs() < 48 * 3600)
    ).astype(int)

    # choose death time if available, otherwise discharge time
    end_event_time = np.where(
        x[C_DOD].notna(),
        x[C_DOD],
        x[C_DISCHTIME]
    )

    x[C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH] = (
        end_event_time - x[C_LAST_TIMESTEP]
    ) / 3600.0

    died_in_icu = x[
        (x[C_DIED_WITHIN_48H_OF_OUT_TIME] == 1) &
        (x[C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH] < 24)
    ][C_ICUSTAYID].unique()

    return set(died_in_icu)


def multiple_icu_stay_ids(demog):
    """
    Return ICU stay IDs belonging to patients with more than one ICU stay.
    """
    needed_cols = [C_SUBJECT_ID, C_ICUSTAYID]
    for col in needed_cols:
        if col not in demog.columns:
            raise ValueError(f"Missing column in demog: {col}")

    demo_small = demog[[C_SUBJECT_ID, C_ICUSTAYID]].dropna().drop_duplicates()

    counts = demo_small.groupby(C_SUBJECT_ID)[C_ICUSTAYID].nunique()
    multi_subjects = set(counts[counts > 1].index)

    remove_ids = set(
        demo_small.loc[demo_small[C_SUBJECT_ID].isin(multi_subjects), C_ICUSTAYID].unique()
    )
    return remove_ids


def build_sepsis_cohort(states, demog):
    """
    Build one-row-per-stay cohort table with severity and outcome.
    """
    agg_dict = {C_SOFA: "max"}
    if C_SIRS in states.columns:
        agg_dict[C_SIRS] = "max"

    state_summary = states.groupby(C_ICUSTAYID).agg(agg_dict).rename(
        columns={
            C_SOFA: C_MAX_SOFA,
            C_SIRS: C_MAX_SIRS
        }
    )

    outcome = demog[[C_ICUSTAYID, C_MORTA_90]].drop_duplicates(subset=C_ICUSTAYID)
    outcome = outcome.set_index(C_ICUSTAYID)

    sepsis = state_summary.join(outcome, how="inner")
    sepsis = sepsis[sepsis[C_MAX_SOFA] >= 2]

    return sepsis


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Build MedDreamer-style sepsis cohort and filter states/actions/mask/delta."
    )

    parser.add_argument("--states", required=True, type=str,
                        help="Path to patient states CSV")
    parser.add_argument("--actions", required=True, type=str,
                        help="Path to actions CSV")
    parser.add_argument("--demog", required=True, type=str,
                        help="Path to demographics CSV")

    parser.add_argument("--mask", type=str, default=None,
                        help="Optional path to mask CSV")
    parser.add_argument("--delta", type=str, default=None,
                        help="Optional path to delta CSV")
    parser.add_argument("--qstime", type=str, default=None,
                        help="Optional path to qstime CSV")

    parser.add_argument("--output-dir", required=True, type=str,
                        help="Directory where filtered outputs will be written")

    # By default, all exclusions are applied.
    # Use --keep-* flags to retain those cases.
    parser.add_argument(
        "--keep-outliers",
        action="store_true",
        help="Keep stays with extreme outlier values (default: excluded)"
    )

    parser.add_argument(
        "--keep-withdrawal-cases",
        action="store_true",
        help="Keep stays matching treatment withdrawal / treatment stopped criteria (default: excluded)"
    )

    parser.add_argument(
        "--keep-missing-static",
        action="store_true",
        help="Keep stays with missing static features across the whole stay (default: excluded)"
    )

    parser.add_argument(
        "--keep-missing-outcome",
        action="store_true",
        help="Keep stays with missing mortality outcome (default: excluded)"
    )

    parser.add_argument(
        "--keep-died-in-icu-during-collection",
        action="store_true",
        help="Keep stays where the patient died in ICU / near ICU outtime during the data collection period (default: excluded)"
    )

    parser.add_argument(
        "--keep-multiple-icu-stays",
        action="store_true",
        help="Keep patients with multiple ICU stays (default: excluded)"
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

    print(f"States stays before filtering: {states[C_ICUSTAYID].nunique()}")
    print(f"Actions stays before filtering: {actions[C_ICUSTAYID].nunique()}")
    print("Filtering configuration:")
    print(f"  Exclude multiple ICU stays: {not args.keep_multiple_icu_stays}")
    print(f"  Exclude outliers: {not args.keep_outliers}")
    print(f"  Exclude withdrawal cases: {not args.keep_withdrawal_cases}")
    print(f"  Exclude missing static: {not args.keep_missing_static}")
    print(f"  Exclude missing outcome: {not args.keep_missing_outcome}")
    print(f"  Exclude died in ICU during collection: {not args.keep_died_in_icu_during_collection}")

    remove_ids = set()

    if not args.keep_multiple_icu_stays:
        multi_ids = multiple_icu_stay_ids(demog)
        print(f"Multiple-ICU-stay stays to remove: {len(multi_ids)}")
        remove_ids |= multi_ids

    if not args.keep_outliers:
        outliers = outlier_stay_ids(states, actions)
        print(f"Outlier stays to remove: {len(outliers)}")
        remove_ids |= outliers

    if not args.keep_withdrawal_cases:
        stopped = treatment_stopped_stay_ids(states, demog, actions)
        print(f"Treatment stopped stays to remove: {len(stopped)}")
        remove_ids |= stopped

    if not args.keep_missing_static:
        print("Checking missing static features...")
        missing_static = missing_static_stay_ids(states)
        print(f"Missing static stays: {len(missing_static)}")
        remove_ids |= missing_static

    if not args.keep_missing_outcome:
        print("Checking missing mortality outcomes...")
        missing_outcome = missing_outcome_stay_ids(demog)
        print(f"Missing mortality stays: {len(missing_outcome)}")
        remove_ids |= missing_outcome

    if not args.keep_died_in_icu_during_collection and qstime is not None:
        died_in_icu = died_in_icu_during_collection_stay_ids(demog, qstime)
        print(f"Died in ICU during collection stays to remove: {len(died_in_icu)}")
        remove_ids |= died_in_icu

    print(f"Total unique stays to remove: {len(remove_ids)}")

    if len(remove_ids) > 0:
        states = states[~states[C_ICUSTAYID].isin(remove_ids)].copy()
        actions = actions[~actions[C_ICUSTAYID].isin(remove_ids)].copy()
        demog = demog[~demog[C_ICUSTAYID].isin(remove_ids)].copy()

        if mask is not None:
            mask = mask[~mask[C_ICUSTAYID].isin(remove_ids)].copy()

        if delta is not None:
            delta = delta[~delta[C_ICUSTAYID].isin(remove_ids)].copy()

        if qstime is not None:
            qstime = qstime[~qstime[C_ICUSTAYID].isin(remove_ids)].copy()

    print("Building sepsis cohort from patient_states + demog...")
    sepsis = build_sepsis_cohort(states, demog)
    print(f"SOFA >= 2 stays: {len(sepsis)}")

    if qstime is not None:
        qstime_idx = qstime.set_index(C_ICUSTAYID, drop=True)

        if C_ONSET_TIME in qstime_idx.columns:
            sepsis = pd.merge(
                sepsis,
                qstime_idx[[C_ONSET_TIME]],
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

    if qstime is not None:
        qstime_f = qstime[qstime[C_ICUSTAYID].isin(keep_ids)].copy()
        qstime_f.to_csv(
            os.path.join(args.output_dir, "qstime_filtered.csv"),
            index=False
        )

    print("Done.")