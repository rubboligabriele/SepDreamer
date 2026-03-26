import pandas as pd
import os
import argparse
from tqdm import tqdm

from src.preprocessing.columns import *
from src.preprocessing.utils import load_csv

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def time_window(df, col, center_time, lower_window, upper_window):
    if df is None or len(df) == 0:
        return df
    return df[(df[col] >= center_time - lower_window) & (df[col] <= center_time + upper_window)]

def _fetch_stay(df, stay_id, stay_col=C_ICUSTAYID):
    if df is None or len(df) == 0:
        return None
    out = df[df[stay_col] == stay_id]
    if len(out) == 0:
        return None
    return out

def _union_timesteps(*series_list):
    series_list = [s for s in series_list if s is not None and len(s) > 0]
    if len(series_list) == 0:
        return []
    return sorted(pd.unique(pd.concat(series_list, ignore_index=True)))

def build_patient_states_derived(
    onset_data,
    demog,
    vital, bg, fio2, cbc, labs, ionca, mag, liver, coag, gcs, urine,
    mechvent, weight, sofa,
    winb4, winaft
):
    combined_data = []
    infection_times = []

    buffer = 4
    bounds = ((winb4 + buffer) * 3600, (winaft + buffer) * 3600)

    # Columns we will copy “as-is” at a given charttime from point-event tables
    point_tables = [
        ("vital",  vital,  C_CHARTTIME),
        ("bg",     bg,     C_CHARTTIME),
        ("fio2",   fio2,   C_CHARTTIME),
        ("cbc",    cbc,    C_CHARTTIME),
        ("labs",   labs,   C_CHARTTIME),
        ("ionca",  ionca,  C_CHARTTIME),
        ("mag",    mag,    C_CHARTTIME),
        ("liver",  liver,  C_CHARTTIME),
        ("coag",   coag,   C_CHARTTIME),
        ("gcs",    gcs,    C_CHARTTIME),
        ("urine",  urine,  C_CHARTTIME),
    ]

    # Interval tables: active if start<=t<=end
    # mechvent_derived has ventilation_status (string) -> we’ll map to 1/0
    # vaso_derived has rate_std
    interval_tables = [
        ("mechvent", mechvent, C_STARTTIME, C_ENDTIME),
        ("weight",   weight,   C_STARTTIME, C_ENDTIME),
        ("sofa",     sofa,     C_STARTTIME, C_ENDTIME),
    ]

    for _, row in tqdm(onset_data.iterrows(), total=len(onset_data), desc="Building patient states (derived)"):
        icustayid = int(row[C_ICUSTAYID])
        qst = row[C_ONSET_TIME]
        if not (qst and qst > 0):
            continue

        d1 = demog.loc[
            demog[C_ICUSTAYID] == icustayid,
            [C_AGE, C_DISCHTIME, C_GENDER, C_ELIXHAUSER, C_ADM_ORDER]
        ]

        if len(d1) == 0:
            continue

        row_demog = d1.iloc[0]

        age = row_demog[C_AGE]
        if age < 18:
            continue

        gender = row_demog[C_GENDER]
        elix = row_demog[C_ELIXHAUSER]
        adm_order = row_demog.get(C_ADM_ORDER, 1)
        readmission = int(adm_order > 1)
        dischtime = row_demog[C_DISCHTIME]

        # --- fetch + window point tables
        stay_point = {}
        for name, df, tcol in point_tables:
            s = _fetch_stay(df, icustayid)
            if s is None:
                stay_point[name] = None
                continue
            s = time_window(s, tcol, qst, *bounds)
            stay_point[name] = s if (s is not None and len(s) > 0) else None

        # --- fetch + window interval tables (window by starttime)
        stay_interval = {}
        lower = qst - bounds[0]
        upper = qst + bounds[1]

        for name, df, scol, ecol in interval_tables:
            s = _fetch_stay(df, icustayid)
            if s is None:
                stay_interval[name] = None
                continue

            # interval intersects [lower, upper] iff end>=lower and start<=upper
            s = s[(s[ecol] >= lower) & (s[scol] <= upper)]
            stay_interval[name] = s if (len(s) > 0) else None

        # --- build timesteps as union of all charttime + starttime of intervals
        ts_series = []
        for name, df, tcol in point_tables:
            s = stay_point.get(name)
            if s is not None and tcol in s.columns:
                ts_series.append(s[tcol])

        for name, df, scol, ecol in interval_tables:
            s = stay_interval.get(name)
            if s is not None and scol in s.columns:
                ts_series.append(s[scol])

        timesteps = _union_timesteps(*ts_series)
        if len(timesteps) == 0:
            continue

        # --- build rows
        for i, t in enumerate(timesteps):
            item = {
                C_BLOC: i,
                C_ICUSTAYID: icustayid,
                C_TIMESTEP: t,
            }

            item[C_AGE] = age
            item[C_GENDER] = gender
            item[C_ELIXHAUSER] = elix
            item["readmission"] = readmission

            # point events: copy all columns except ids/time
            for name, _, tcol in point_tables:
                s = stay_point.get(name)
                if s is None:
                    continue
                rows = s[s[tcol] == t]
                if len(rows) == 0:
                    continue

                # if duplicates at same timestamp, take last non-null per column
                r = rows.sort_index().iloc[-1]
                for col in rows.columns:
                    if col in (C_ICUSTAYID, tcol):
                        continue
                    item[col] = r[col]

            # mechvent interval -> mechvent=1 if ventilation_status indicates ventilation
            item[C_MECHVENT] = pd.NA
            mv = stay_interval.get("mechvent")
            if mv is not None and len(mv) > 0:
                active = mv[(mv[C_STARTTIME] <= t) & (t <= mv[C_ENDTIME])]
                if len(active) > 0:
                    status = active.iloc[-1].get("mechvent", active.iloc[-1].get("ventilation_status", None))
                    if pd.isna(status) or status is None:
                        item[C_MECHVENT] = pd.NA
                    else:
                        item[C_MECHVENT] = int(str(status).lower() != "none")

            # weight interval
            wt = stay_interval.get("weight")
            if wt is not None and len(wt) > 0:
                active = wt[(wt[C_STARTTIME] <= t) & (t <= wt[C_ENDTIME])]
                if len(active) > 0:
                    item[C_WEIGHT] = active.iloc[-1].get(C_WEIGHT, active.iloc[-1].get("Weight_kg", pd.NA))

            # sofa interval
            sf = stay_interval.get("sofa")
            if sf is not None and len(sf) > 0:
                active = sf[(sf[C_STARTTIME] <= t) & (t <= sf[C_ENDTIME])]
                if len(active) > 0:
                    item[C_SOFA] = active.iloc[-1].get(C_SOFA, active.iloc[-1].get("sofa", pd.NA))

            combined_data.append(item)

        infection_times.append({
            C_ICUSTAYID: icustayid,
            C_ONSET_TIME: qst,
            C_FIRST_TIMESTEP: timesteps[0],
            C_LAST_TIMESTEP: timesteps[-1],
            C_DISCHTIME: dischtime,
        })

    state_df = pd.DataFrame(combined_data)
    qstime = pd.DataFrame(infection_times).set_index(C_ICUSTAYID)

    return state_df, qstime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        "Builds patient states from MIMIC-IV derived tables (MedDreamer style): "
        "no ce/labs itemid mapping, no binning."
    ))
    parser.add_argument("output_dir", type=str,
                        help="Directory in which to output (e.g. data/intermediates/patient_states)")
    parser.add_argument("--data", dest="data_dir", type=str, default=None,
                        help="Data directory (default: ../data)")
    parser.add_argument("--window-before", dest="window_before", type=int, default=25)
    parser.add_argument("--window-after", dest="window_after", type=int, default=49)
    parser.add_argument("--head", dest="head", type=int, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(PARENT_DIR, "data")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    interm = os.path.join(data_dir, "intermediates")

    print("Reading onset (derived)...")
    onset = load_csv(os.path.join(interm, "onset_derived.csv"))
    # Expect columns: stay_id, subject_id, sepsis_onset_time (UNIX seconds or datetime->must be UNIX)
    # You want to standardize to:
    #   C_ICUSTAYID, C_SUBJECT_ID, C_ONSET_TIME
    rename_map = {
        "stay_id": C_ICUSTAYID,
        "sepsis_onset_time": C_ONSET_TIME,
    }
    onset = onset.rename(columns=rename_map)
    if args.head:
        onset = onset.head(args.head)

    print("Reading demog...")
    demog = load_csv(os.path.join(interm, "demog.csv"))

    def r(name):
        path = os.path.join(interm, f"{name}.csv")
        return load_csv(path) if os.path.exists(path) else pd.DataFrame()

    print("Reading derived tables...")
    vital   = r("vital_derived")
    bg      = r("bg_derived")
    fio2    = r("fio2_derived")
    cbc     = r("cbc_derived")
    labs    = r("labs_derived")
    ionca   = r("ion_cal_derived")
    mag     = r("mag_derived")
    liver   = r("liver_derived")
    coag    = r("coag_derived")
    gcs     = r("gcs_derived")
    urine   = r("urine_derived")
    mechvent = r("mechvent_derived")
    weight  = r("weight_derived")
    sofa    = r("sofa_derived")

    state_df, qstime = build_patient_states_derived(
        onset_data=onset,
        demog=demog,
        vital=vital, bg=bg, fio2=fio2, cbc=cbc, labs=labs, ionca=ionca, mag=mag,
        liver=liver, coag=coag, gcs=gcs, urine=urine,
        mechvent=mechvent, weight=weight, sofa=sofa,
        winb4=args.window_before,
        winaft=args.window_after
    )

    print(f"Result: state_df={len(state_df)} rows, {len(state_df.columns)} cols")

    state_df.to_csv(os.path.join(out_dir, "patient_states.csv"), index=False, float_format="%g")
    qstime.to_csv(os.path.join(out_dir, "qstime.csv"), float_format="%g")