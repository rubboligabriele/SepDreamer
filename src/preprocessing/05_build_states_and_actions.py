import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing.columns import *
from src.preprocessing.utils import load_csv

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def _sum_fluid_interval_mv(fluid_df, t0, t1):
    """
    Fluid delivered in [t0, t1) from MetaVision, using the same logic as AI Clinician:
    - infusions: rate * temporal overlap
    - boluses: tev if starttime falls inside the interval
    """
    if fluid_df is None or len(fluid_df) == 0:
        return 0.0

    total = 0.0

    # Infusions: rows with non-null rate
    inf = fluid_df[~pd.isna(fluid_df[C_RATE])]
    if len(inf) > 0:
        startt = inf[C_STARTTIME].to_numpy(dtype=float)
        endt = inf[C_ENDTIME].to_numpy(dtype=float)
        rate = inf[C_RATE].to_numpy(dtype=float)

        total += np.nansum(
            rate * (endt - startt) * ((endt <= t1) & (startt >= t0)) / 3600.0 +
            rate * (endt - t0) * ((startt <= t0) & (endt <= t1) & (endt >= t0)) / 3600.0 +
            rate * (t1 - startt) * ((startt >= t0) & (endt >= t1) & (startt <= t1)) / 3600.0 +
            rate * (t1 - t0) * ((endt >= t1) & (startt <= t0)) / 3600.0
        )

    # Boluses: rows with null rate, counted by starttime inside interval
    bol = fluid_df[pd.isna(fluid_df[C_RATE])]
    if len(bol) > 0:
        mask = (bol[C_STARTTIME] >= t0) & (bol[C_STARTTIME] <= t1)
        if mask.any():
            total += bol.loc[mask, C_TEV].fillna(0).sum()

    return float(total)


def _sum_uo_interval(uo_df, t0, t1):
    if uo_df is None or len(uo_df) == 0:
        return 0.0

    mask = (uo_df[C_CHARTTIME] >= t0) & (uo_df[C_CHARTTIME] <= t1)
    if not mask.any():
        return 0.0

    # support both formats
    if C_VALUE in uo_df.columns:
        return float(uo_df.loc[mask, C_VALUE].fillna(0).sum())

    if "output_total" in uo_df.columns:
        return float(uo_df.loc[mask, "output_total"].fillna(0).sum())

    raise ValueError("Urine output column not found")


def _vaso_stats_interval(vaso_df, t0, t1):
    """
    Vasopressor stats inside [t0, t1], using vaso_derived:
    active if infusion interval overlaps [t0, t1]
    """
    if vaso_df is None or len(vaso_df) == 0:
        return np.nan, np.nan

    startv = vaso_df[C_STARTTIME]
    endv = vaso_df[C_ENDTIME]
    ratev = vaso_df[C_RATESTD]

    v = (
        ((endv >= t0) & (endv <= t1)) |
        ((startv >= t0) & (endv <= t1)) |
        ((startv >= t0) & (startv <= t1)) |
        ((startv <= t0) & (endv >= t1))
    )

    vals = ratev.loc[v].dropna().to_numpy(dtype=float)

    if len(vals) == 0:
        return np.nan, np.nan

    return float(np.nanmedian(vals)), float(np.nanmax(vals))


def _compute_prestate_fluid_total(inputMV, inputpreadm, stay_id, first_t):
    """
    Total fluid before first patient state timestep:
    - preadmission fluid
    - MV fluid from [0, first_t]
    """
    totvol = 0.0

    if inputpreadm is not None and len(inputpreadm) > 0:
        pread = inputpreadm.loc[inputpreadm[C_ICUSTAYID] == stay_id, C_INPUT_PREADM]
        if not pread.empty:
            totvol += float(pread.fillna(0).sum())

    if inputMV is not None and len(inputMV) > 0:
        totvol += _sum_fluid_interval_mv(inputMV, 0, first_t)

    return float(totvol)


def _compute_prestate_uo_total(UO, UOpreadm, stay_id, first_t):
    """
    Total urine output before first patient state timestep.
    """
    UOtot = 0.0

    if UOpreadm is not None and len(UOpreadm) > 0:
        pread = UOpreadm.loc[UOpreadm[C_ICUSTAYID] == stay_id, C_VALUE]
        if not pread.empty:
            UOtot += float(pread.fillna(0).sum())

    if UO is not None and len(UO) > 0:
        UOtot += _sum_uo_interval(UO, 0, first_t)

    return float(UOtot)


def build_states_and_actions_irregular(
    df,
    inputMV,
    inputpreadm,
    vaso,
    UOpreadm,
    UO,
    head=None,
    allowed_stays=None
):
    """
    Adapts AI Clinician state/action construction to MedDreamer setting:
    - NO temporal binning
    - one action row per irregular interval [t_i, t_{i+1}]
    - continuous actions only
    """
    icustayidlist = np.unique(df[C_ICUSTAYID])
    icustayidlist = sorted(icustayidlist[~pd.isna(icustayidlist)])

    if allowed_stays is not None:
        old_count = len(icustayidlist)
        allowed_stays = set(allowed_stays)
        icustayidlist = [sid for sid in icustayidlist if sid in allowed_stays]
        print(f"Filtered from {old_count} to {len(icustayidlist)} ICU stay ids")
    else:
        print(f"{len(icustayidlist)} ICU stay IDs")

    if head:
        icustayidlist = icustayidlist[:head]

    combined_data = []

    for icustayid in tqdm(icustayidlist, desc="Building irregular states and actions"):
        temp = df.loc[df[C_ICUSTAYID] == icustayid, :].copy()
        temp = temp.sort_values(C_TIMESTEP).reset_index(drop=True)

        if len(temp) < 2:
            continue

        beg = float(temp[C_TIMESTEP].iloc[0])

        input1 = inputMV.loc[inputMV[C_ICUSTAYID] == icustayid, :] if inputMV is not None else None
        totvol = _compute_prestate_fluid_total(input1, inputpreadm, icustayid, beg)

        # Vasopressors
        vaso1 = vaso.loc[vaso[C_ICUSTAYID] == icustayid, :] if vaso is not None else None

        # Urine output
        output = UO.loc[UO[C_ICUSTAYID] == icustayid, :] if UO is not None else None
        UOtot = _compute_prestate_uo_total(output, UOpreadm, icustayid, beg)

        # one row per interval [t_i, t_{i+1}]
        for i in range(len(temp) - 1):
            t0 = float(temp.loc[i, C_TIMESTEP])
            t1 = float(temp.loc[i + 1, C_TIMESTEP])

            if t1 <= t0:
                continue

            item = {
                C_BLOC: temp.loc[i, C_BLOC],
                C_ICUSTAYID: icustayid,
                C_TIMESTEP: int(t0),
            }

            # VASOPRESSORS
            median_dose_vaso, max_dose_vaso = _vaso_stats_interval(vaso1, t0, t1)
            item[C_MEDIAN_DOSE_VASO] = median_dose_vaso
            item[C_MAX_DOSE_VASO] = max_dose_vaso

            # INPUT FLUID
            fluid_step = _sum_fluid_interval_mv(input1, t0, t1)

            totvol = np.nansum([totvol, fluid_step])
            item[C_INPUT_TOTAL] = float(totvol)
            item[C_INPUT_STEP] = float(fluid_step)

            # UO
            UOnow = _sum_uo_interval(output, t0, t1)
            UOtot = np.nansum([UOtot, UOnow])
            item[C_OUTPUT_TOTAL] = float(UOtot)
            item[C_OUTPUT_STEP] = float(UOnow)

            # CUMULATED BALANCE
            item[C_CUMULATED_BALANCE] = float(totvol - UOtot)

            combined_data.append(item)

    result = pd.DataFrame(combined_data)

    expected_columns = IO_FIELD_NAMES
    for col in expected_columns:
        if col not in result.columns:
            print(f"Adding empty column '{col}' (no data points)")
            result[col] = pd.NA

    # --- AI Clinician / MedDreamer behaviour ---
    # If no vasopressor in interval → dose = 0 (not NaN)

    if C_MEDIAN_DOSE_VASO in result.columns:
        result[C_MEDIAN_DOSE_VASO] = result[C_MEDIAN_DOSE_VASO].fillna(0)

    if C_MAX_DOSE_VASO in result.columns:
        result[C_MAX_DOSE_VASO] = result[C_MAX_DOSE_VASO].fillna(0)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build irregular action/I-O table from patient_states timesteps "
            "(MedDreamer style, no temporal binning)."
        )
    )
    parser.add_argument("input", type=str, help="Patient states file")
    parser.add_argument("output", type=str, help="CSV path to write output")
    parser.add_argument("--data", dest="data_dir", type=str, default=None,
                        help="Directory in which raw and preprocessed data is stored (default: ../data)")
    parser.add_argument("--head", dest="head", type=int, default=None,
                        help="Number of ICU stays to convert")
    parser.add_argument("--filter-stays", dest="filter_stays_path", type=str, default=None,
                        help="Path to a CSV file containing an icustayid column")

    args = parser.parse_args()
    data_dir = args.data_dir or os.path.join(PARENT_DIR, "data")
    interm_dir = os.path.join(data_dir, "intermediates")
    raw_dir = os.path.join(data_dir, "raw_data")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    def load_intermediate_or_raw_csv(filename):
        p1 = os.path.join(interm_dir, filename)
        p2 = os.path.join(raw_dir, filename)
        if os.path.exists(p1):
            return load_csv(p1)
        if os.path.exists(p2):
            return load_csv(p2)
        return None

    print("Reading states...")
    df = load_csv(args.input)

    print("Reading data files...")
    inputpreadm = load_intermediate_or_raw_csv("preadm_fluid.csv")
    inputMV = load_intermediate_or_raw_csv("fluid_mv.csv")
    vaso = load_intermediate_or_raw_csv("vaso_derived.csv")
    UOpreadm = load_intermediate_or_raw_csv("preadm_uo.csv")
    UO = load_intermediate_or_raw_csv("urine_derived.csv")

    if inputMV is not None and "icustay_id" in inputMV.columns:
        inputMV = inputMV.rename(columns={"icustay_id": C_ICUSTAYID})

    allowed_stays = None
    if args.filter_stays_path:
        print("Reading filter stays...")
        allowed_stays_df = load_csv(args.filter_stays_path)
        allowed_stays = allowed_stays_df[C_ICUSTAYID]

    result = build_states_and_actions_irregular(
        df=df,
        inputMV=inputMV,
        inputpreadm=inputpreadm,
        vaso=vaso,
        UOpreadm=UOpreadm,
        UO=UO,
        head=args.head,
        allowed_stays=allowed_stays,
    )

    print("Writing to file")
    result.to_csv(args.output, index=False, float_format="%g")