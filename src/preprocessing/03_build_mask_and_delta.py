import pandas as pd
import numpy as np
import os
import argparse

from src.preprocessing.provenance import ProvenanceWriter
from src.preprocessing.columns import *
from src.preprocessing.utils import load_csv
from src.preprocessing.imputation import fill_outliers
from src.preprocessing.derived_features import compute_shock_index, compute_sirs

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def correct_features(df, provenance=None):
    # Gender: from 1/2 to 0/1
    if C_GENDER in df.columns:
        df[C_GENDER] = df[C_GENDER] - 1

    # Age clamp
    if C_AGE in df.columns:
        ii = df[C_AGE] > 150
        if provenance:
            provenance.record("clamp age", row=df.loc[ii].index, col=C_AGE)
        df.loc[ii, C_AGE] = 91

    return df

def remove_outliers(df, provenance=None):
    def has(col):
        return col in df.columns

    if has(C_TEMP_C) and has(C_TEMP_F):
        wrong_unit_temps = (df[C_TEMP_C] > 90) & pd.isna(df[C_TEMP_F])
        if provenance:
            provenance.record(
                "temp_F logged as temp_C",
                row=df.loc[wrong_unit_temps].index,
                col=C_TEMP_F,
                reference_col=C_TEMP_C
            )
        df.loc[wrong_unit_temps, C_TEMP_F] = df.loc[wrong_unit_temps, C_TEMP_C]

    if has(C_FIO2_100):
        ii = df[C_FIO2_100] < 1
        if provenance:
            provenance.record("FiO2_100 < 1 => *100", row=df.loc[ii].index, col=C_FIO2_100)
        df.loc[ii, C_FIO2_100] = df.loc[ii, C_FIO2_100] * 100

    bounds = {}

    # weight
    if has(C_WEIGHT): bounds[C_WEIGHT] = (None, 300)

    # HR
    if has(C_HR): bounds[C_HR] = (None, 250)

    # BP
    if has(C_SYSBP):  bounds[C_SYSBP]  = (None, 300)
    if has(C_MEANBP): bounds[C_MEANBP] = (0, 200)
    if has(C_DIABP):  bounds[C_DIABP]  = (0, 200)

    # RR
    if has(C_RR): bounds[C_RR] = (None, 80)

    # SpO2
    if has(C_SPO2): bounds[C_SPO2] = (None, 150)

    # Temp_C
    if has(C_TEMP_C): bounds[C_TEMP_C] = (None, 90)

    # FiO2
    if has(C_FIO2_100): bounds[C_FIO2_100] = (20, 100)
    if has(C_FIO2_1):   bounds[C_FIO2_1]   = (None, 1.5)

    # O2 FLOW
    if has(C_O2FLOW): bounds[C_O2FLOW] = (None, 70)

    # PEEP
    if has(C_PEEP): bounds[C_PEEP] = (0, 40)

    # TV / MV
    if has(C_TIDALVOLUME): bounds[C_TIDALVOLUME] = (None, 1800)
    if has(C_MINUTEVENTIL): bounds[C_MINUTEVENTIL] = (None, 50)

    for col, lo, hi in [
        (C_POTASSIUM, 1, 15),
        (C_SODIUM, 95, 178),
        (C_CHLORIDE, 70, 150),
        (C_GLUCOSE, 1, 1000),
        (C_CREATININE, None, 150),
        (C_MAGNESIUM, None, 10),
        (C_CALCIUM, None, 20),
        (C_IONISED_CA, None, 5),
        (C_CO2_MEQL, None, 120),
        (C_SGPT, None, 10000),
        (C_SGOT, None, 10000),
        (C_HB, None, 20),
        (C_HT, None, 65),
        (C_WBC_COUNT, None, 500),
        (C_PLATELETS_COUNT, None, 2000),
        (C_INR, None, 20),
        (C_ARTERIAL_PH, 6.7, 8),
        (C_PAO2, None, 700),
        (C_PACO2, None, 200),
        (C_ARTERIAL_BE, -50, None),
        (C_ARTERIAL_LACTATE, None, 30),
    ]:
        if has(col):
            bounds[col] = (lo, hi)

    df = fill_outliers(df, bounds, provenance=provenance)

    if has(C_SPO2):
        if provenance:
            provenance.record("Clamp SpO2 > 100", row=df.loc[df[C_SPO2] > 100].index, col=C_SPO2)
        df.loc[df[C_SPO2] > 100, C_SPO2] = 100
        df = fill_outliers(df, {C_SPO2: (50, None)}, provenance=provenance)

    if has(C_TEMP_C):
        df = fill_outliers(df, {C_TEMP_C: (25, None)}, provenance=provenance)

    return df


def compute_mask_delta(df, feature_cols, id_cols, observed_mask=None):
    df = df.sort_values([C_ICUSTAYID, C_TIMESTEP]).reset_index(drop=True)

    static_cols = [
        C_AGE,
        C_GENDER,
        C_ELIXHAUSER,
        C_RE_ADMISSION,
        C_WEIGHT,
    ]
    static_cols = [c for c in static_cols if c in feature_cols]
    dynamic_cols = [c for c in feature_cols if c not in static_cols]

    mask_df = df[id_cols].copy()

    if observed_mask is None:
        if dynamic_cols:
            mask_df[dynamic_cols] = (~df[dynamic_cols].isna()).astype(np.float32)
    else:
        if dynamic_cols:
            mask_df[dynamic_cols] = observed_mask[dynamic_cols].astype(np.float32).to_numpy()

    if static_cols:
        mask_df[static_cols] = 1.0

    mask_df = mask_df[id_cols + feature_cols]

    delta_df = df[id_cols].copy()
    delta_vals = np.zeros((len(df), len(feature_cols)), dtype=np.float32)

    feature_to_idx = {c: i for i, c in enumerate(feature_cols)}
    dyn_idx = [feature_to_idx[c] for c in dynamic_cols]
    stat_idx = [feature_to_idx[c] for c in static_cols]

    for _, g in df.groupby(C_ICUSTAYID, sort=False):
        idx = g.index.to_numpy()
        t = g[C_TIMESTEP].to_numpy(dtype=np.float64)

        d = np.zeros((len(idx), len(feature_cols)), dtype=np.float64)

        if stat_idx:
            d[:, stat_idx] = 0.0

        if dyn_idx:
            m_dyn = mask_df.loc[idx, dynamic_cols].to_numpy(dtype=np.float32)

            for k in range(1, len(idx)):
                dt = max(0.0, t[k] - t[k - 1]) / 3600.0  # hours, come medR

                d[k, dyn_idx] = np.where(
                    m_dyn[k - 1, :] == 1.0,
                    dt,
                    dt + d[k - 1, dyn_idx]
                )

        delta_vals[idx, :] = d.astype(np.float32)

    delta_df[feature_cols] = delta_vals
    delta_df = delta_df[id_cols + feature_cols]

    return mask_df, delta_df


def compute_freshness_delta(df, feature_cols, id_cols, observed_mask=None):
    df = df.sort_values([C_ICUSTAYID, C_TIMESTEP]).reset_index(drop=True)

    static_cols = [C_AGE, C_GENDER, C_ELIXHAUSER, C_RE_ADMISSION, C_WEIGHT]
    static_cols = [c for c in static_cols if c in feature_cols]
    dynamic_cols = [c for c in feature_cols if c not in static_cols]

    delta_df = df[id_cols].copy()
    delta_vals = np.zeros((len(df), len(feature_cols)), dtype=np.float32)

    feature_to_idx = {c: i for i, c in enumerate(feature_cols)}
    dyn_idx = [feature_to_idx[c] for c in dynamic_cols]
    stat_idx = [feature_to_idx[c] for c in static_cols]

    if observed_mask is None:
        observed_mask = df[dynamic_cols].notna().astype(np.float32)

    for _, g in df.groupby(C_ICUSTAYID, sort=False):
        idx = g.index.to_numpy()
        t = g[C_TIMESTEP].to_numpy(dtype=np.float64)

        d = np.zeros((len(idx), len(feature_cols)), dtype=np.float64)

        if stat_idx:
            d[:, stat_idx] = 0.0

        if dyn_idx:
            m_dyn = observed_mask.loc[idx, dynamic_cols].to_numpy(dtype=np.float32)

            for k in range(1, len(idx)):
                dt = max(0.0, t[k] - t[k - 1]) / 3600.0

                d[k, dyn_idx] = np.where(
                    m_dyn[k, :] == 1.0,
                    0.0,
                    d[k - 1, dyn_idx] + dt
                )

        delta_vals[idx, :] = d.astype(np.float32)

    delta_df[feature_cols] = delta_vals
    return delta_df[id_cols + feature_cols]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Patient states postprocess (MedDreamer-style): outliers, derived features, mask & delta."
    )
    parser.add_argument("input", type=str, help="Path to patient states CSV file")
    parser.add_argument("output", type=str, help="Path at which to write output")

    parser.add_argument("--no-outliers", dest="outliers", default=True, action="store_false",
                        help="Don't replace outliers with NaNs")

    # Derived features
    parser.add_argument("--shock-index", dest="shock_index", default=True, action="store_true",
                        help="Compute shock index column")
    parser.add_argument("--no-shock-index", dest="shock_index", action="store_false",
                        help="Don't compute shock index")
    parser.add_argument("--sirs", dest="sirs", default=True, action="store_true",
                        help="Compute SIRS column")
    parser.add_argument("--no-sirs", dest="sirs", action="store_false",
                        help="Don't compute SIRS")

    # Outputs
    parser.add_argument("--mask-out", dest="mask_out", default=None, type=str,
                        help="Path to write MedDreamer mask (1=observed,0=missing)")
    parser.add_argument("--delta-out", dest="delta_out", default=None, type=str,
                        help="Path to write MedDreamer delta (time since last observed per feature)")
    parser.add_argument("--delta-fresh-out", dest="delta_fresh_out", default=None, type=str,
                    help="Path to write freshness delta for medR reward computation")
    parser.add_argument("--mask-file", dest="mask_file", default=None, type=str,
                        help="Diff mask file (+1 added/changed, -1 removed, 0 unchanged)")
    parser.add_argument("--provenance-dir", dest="provenance_dir", default=None, type=str,
                        help="Directory to write provenance logs")

    args = parser.parse_args()

    df = load_csv(args.input)
    old_df = df.copy() if args.mask_file else None

    provenance = ProvenanceWriter(args.provenance_dir, verbose=True) if args.provenance_dir else None

    # ---- cleaning
    if args.outliers:
        print("Remove outliers")
        df = remove_outliers(df, provenance=provenance)

    print("Correct features")
    df = correct_features(df, provenance=provenance)

    if args.shock_index:
        print("Computing shock index")
        df[C_SHOCK_INDEX] = compute_shock_index(df)

    if args.sirs:
        print("Computing SIRS")
        df[C_SIRS] = compute_sirs(df)

    # ---- choose feature cols for mask/delta (include demog + chart + labs + mechvent/extubated + derived)
    ID_COLS = [C_BLOC, C_ICUSTAYID, C_TIMESTEP]
    feature_cols = [c for c in df.columns if c not in ID_COLS]

    df = df.sort_values([C_ICUSTAYID, C_TIMESTEP]).reset_index(drop=True)

    # ---- raw observation mask BEFORE forward fill
    static_cols = [
        C_AGE,
        C_GENDER,
        C_ELIXHAUSER,
        C_RE_ADMISSION,
        C_WEIGHT,
    ]
    static_cols = [c for c in static_cols if c in feature_cols]
    dynamic_cols = [c for c in feature_cols if c not in static_cols]

    observed_mask = df[dynamic_cols].notna().astype(np.float32)

    # ---- mask/delta from original missingness
    mask_df, delta_df = compute_mask_delta(
        df,
        feature_cols=feature_cols,
        id_cols=ID_COLS,
        observed_mask=observed_mask,
    )

    delta_fresh_df = compute_freshness_delta(
        df,
        feature_cols=feature_cols,
        id_cols=ID_COLS,
        observed_mask=observed_mask,
    )

    assert list(delta_df.columns) == list(delta_fresh_df.columns)
    assert len(delta_df) == len(delta_fresh_df)

    # ---- forward fill values AFTER mask/delta
    print("Forward filling dynamic features")
    df = df.sort_values([C_ICUSTAYID, C_TIMESTEP]).reset_index(drop=True)

    if dynamic_cols:
        df[dynamic_cols] = (
            df.groupby(C_ICUSTAYID, sort=False)[dynamic_cols]
            .ffill()
        )

    # static columns should already be filled by construction, but this is safe
    if static_cols:
        df[static_cols] = (
            df.groupby(C_ICUSTAYID, sort=False)[static_cols]
            .ffill()
            .bfill()
        )

    # ---- write
    print("Write output")
    df.to_csv(args.output, index=False, float_format="%g")

    if args.mask_out:
        mask_df.to_csv(args.mask_out, index=False, float_format="%g")

    if args.delta_out:
        delta_df.to_csv(args.delta_out, index=False, float_format="%g")

    if args.delta_fresh_out:
        delta_fresh_df.to_csv(args.delta_fresh_out, index=False, float_format="%g")

    if provenance:
        provenance.close()

    # ---- diff mask file (optional)
    if args.mask_file:
        assert old_df is not None
        print("Write diff mask file")
        mask_series = {C_BLOC: df[C_BLOC], C_ICUSTAYID: df[C_ICUSTAYID], C_TIMESTEP: df[C_TIMESTEP]}
        for col in set(old_df.columns) & set(df.columns):
            if col in mask_series:
                continue
            old = old_df[col]
            new = df[col]
            mask_series[col] = np.where(
                (pd.isna(old) & ~pd.isna(new)) | (~pd.isna(old) & (old != new)),
                1,
                np.where(~pd.isna(old) & pd.isna(new), -1, 0)
            )
        pd.DataFrame(mask_series).to_csv(args.mask_file, index=False)