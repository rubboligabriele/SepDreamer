import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

from src.preprocessing.columns import *
from src.preprocessing.utils import load_csv

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

class ChartEvents:
    """
    Manages a set of preprocessed chart event tables (ce*.csv in data/intermediates),
    and supports retrieving all chart events for a given ICU stay ID.
    """
    def __init__(self, ce_dfs, stay_id_col=C_ICUSTAYID):
        super().__init__()
        self.dfs = ce_dfs
        self.ranges = [(df[stay_id_col].min(), df[stay_id_col].max())
                       for df in ce_dfs]
        self.stay_id_col = stay_id_col

    def fetch(self, stay_id):
        results = []
        for df, (min_id, max_id) in zip(self.dfs, self.ranges):
            if stay_id >= min_id and stay_id <= max_id:
                chunk = df[df[self.stay_id_col] == stay_id]
                if len(chunk) > 0:
                    results.append(chunk)
        if results:
            return pd.concat(results, ignore_index=True)
        return None


def time_window(df, col, center_time, lower_window, upper_window):
    """
    Returns all rows in df whose timestamp in column 'col' fall
    between center_time - lower_window and center_time + upper_window.
    """
    if df is None or len(df) == 0:
        return df
    return df[(df[col] >= center_time - lower_window) & (df[col] <= center_time + upper_window)]


def build_patient_states(chart_events, onset_data, demog, labU, MV_procedure, winb4, winaft):
    """
    Builds the patient states dataframe without imputation. Returns:
      - state_df: patient states at each observed timestep (no binning)
      - qstime: per-stay onset/first/last/disch times
    """
    combined_data = []
    infection_times = []
    from_pe = 0  # kept for compatibility / future logging

    # Ensure labU has the minimal columns we access even if empty
    if labU is None or len(labU) == 0:
        labU = pd.DataFrame(columns=[C_ICUSTAYID, C_CHARTTIME, C_ITEMID, C_VALUENUM])

    for _, row in tqdm(onset_data.iterrows(), total=len(onset_data), desc='Building patient states'):
        qst = row[C_ONSET_TIME]
        icustayid = int(row[C_ICUSTAYID])
        if not (qst and qst > 0):
            continue

        d1 = demog.loc[demog[C_ICUSTAYID] == icustayid, [C_AGE, C_DISCHTIME]].values.tolist()
        if len(d1) == 0:
            continue
        if d1[0][0] < 18:
            continue

        # Window: (-winb4-4h, +winaft+4h) around qst
        bounds = ((winb4 + 4) * 3600, (winaft + 4) * 3600)

        # --- CHARTEVENTS (preprocessed) ---
        ce_df = chart_events.fetch(icustayid)
        if ce_df is None or len(ce_df) == 0:
            temp = pd.DataFrame(columns=[C_CHARTTIME, C_ITEMID, C_VALUENUM])
        else:
            temp = time_window(ce_df, C_CHARTTIME, qst, *bounds)
            if temp is None or len(temp) == 0:
                temp = pd.DataFrame(columns=[C_CHARTTIME, C_ITEMID, C_VALUENUM])

        # --- LABEVENTS (preprocessed) ---
        temp2 = labU[labU[C_ICUSTAYID] == icustayid]
        temp2 = time_window(temp2, C_CHARTTIME, qst, *bounds)
        if temp2 is None or len(temp2) == 0:
            temp2 = pd.DataFrame(columns=[C_CHARTTIME, C_ITEMID, C_VALUENUM])

        # --- MECHVENT (procedureevents-derived, already “interval-ish”) ---
        if MV_procedure is not None and len(MV_procedure) > 0:
            temp4 = MV_procedure[MV_procedure[C_ICUSTAYID] == icustayid]
            temp4 = time_window(temp4, C_STARTTIME, qst, *bounds)
            if temp4 is None or len(temp4) == 0:
                temp4 = None
        else:
            temp4 = None

        # --- Build timesteps as union of available timestamps (no binning) ---
        series_list = []
        if len(temp) > 0 and C_CHARTTIME in temp.columns:
            series_list.append(temp[C_CHARTTIME])
        if len(temp2) > 0 and C_CHARTTIME in temp2.columns:
            series_list.append(temp2[C_CHARTTIME])
        if temp4 is not None and len(temp4) > 0 and C_STARTTIME in temp4.columns:
            series_list.append(temp4[C_STARTTIME])

        if len(series_list) == 0:
            continue

        timesteps = sorted(pd.unique(pd.concat(series_list, ignore_index=True)))
        if len(timesteps) == 0:
            continue

        # --- Per timestep: fill vitals/labs by preprocessed “compact” itemid ---
        for i, timestep in enumerate(timesteps):
            item = {
                C_BLOC: i,
                C_ICUSTAYID: icustayid,
                C_TIMESTEP: timestep
            }

            # CHARTEVENTS
            if len(temp) > 0:
                for _, event in temp[temp[C_CHARTTIME] == timestep].iterrows():
                    iid = int(event[C_ITEMID]) if not pd.isna(event[C_ITEMID]) else -1
                    if iid <= 0 or iid > len(CHART_FIELD_NAMES):
                        continue
                    item[CHART_FIELD_NAMES[iid - 1]] = event[C_VALUENUM]

            # LABS
            if len(temp2) > 0:
                for _, event in temp2[temp2[C_CHARTTIME] == timestep].iterrows():
                    iid = int(event[C_ITEMID]) if not pd.isna(event[C_ITEMID]) else -1
                    if iid <= 0 or iid > len(LAB_FIELD_NAMES):
                        continue
                    item[LAB_FIELD_NAMES[iid - 1]] = event[C_VALUENUM]

            # MV flags (default NaN)
            item[C_MECHVENT] = np.nan
            item[C_EXTUBATED] = np.nan
            item[C_SELFEXTUBATED] = np.nan

            if temp4 is not None and len(temp4) > 0:
                # “Active ventilation interval” check
                active = temp4[(temp4[C_STARTTIME] <= timestep) & (timestep <= temp4[C_ENDTIME])]
                if len(active) > 0:
                    item[C_MECHVENT] = int(active[C_MECHVENT].any())
                else:
                    item[C_MECHVENT] = np.nan

                # Extubation events at exact starttime
                ext_evt = temp4[(temp4[C_STARTTIME] == timestep) & (temp4[C_EXTUBATED] == 1)]
                if len(ext_evt) > 0:
                    item[C_EXTUBATED] = 1
                    item[C_SELFEXTUBATED] = int(ext_evt[C_SELFEXTUBATED].any())
                else:
                    item[C_EXTUBATED] = 0
                    item[C_SELFEXTUBATED] = 0

            combined_data.append(item)

        infection_times.append({
            C_ICUSTAYID: icustayid,
            C_ONSET_TIME: qst,
            C_FIRST_TIMESTEP: timesteps[0],
            C_LAST_TIMESTEP: timesteps[-1],
            C_DISCHTIME: d1[0][1]
        })

    print("Got {} items from procedure events".format(from_pe))
    state_df = pd.DataFrame(combined_data)
    qstime = pd.DataFrame(infection_times).set_index(C_ICUSTAYID)

    # Ensure all expected vital/lab columns exist (even if never observed)
    expected_columns = CHART_FIELD_NAMES + LAB_FIELD_NAMES
    for col in expected_columns:
        if col not in state_df.columns:
            state_df[col] = pd.NA

    # Ensure MV columns exist
    for col in [C_MECHVENT, C_EXTUBATED, C_SELFEXTUBATED]:
        if col not in state_df.columns:
            state_df[col] = pd.NA

    return state_df, qstime


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Generates a preliminary CSV containing patient state information at each timestep '
        'for which data is available (no binning). Also generates a dataframe of relevant '
        'timestamps for each patient (qstime).'
    ))
    parser.add_argument('output_dir', type=str,
                        help='Directory in which to output (e.g. data/intermediates/patient_states)')
    parser.add_argument('--data', dest='data_dir', type=str, default=None,
                        help='Directory in which raw and preprocessed data is stored (default is ../data/ directory)')
    parser.add_argument('--window-before', dest='window_before', type=int, default=24,
                        help="Number of hours before sepsis onset to include data (default 24)")
    parser.add_argument('--window-after', dest='window_after', type=int, default=48,
                        help="Number of hours after sepsis onset to include data (default 48)")
    parser.add_argument('--head', dest='head', type=int, default=None,
                        help='Number of rows at the beginning of onset data to convert to patient states')
    parser.add_argument('--filter-stays', dest='filter_stays_path', type=str, default=None,
                        help='Path to a CSV file containing an icustayid column; output will be filtered to these ICU stays')

    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(PARENT_DIR, 'data')
    out_dir = args.output_dir or os.path.join(PARENT_DIR, 'data', 'intermediates', 'patient_states')
    os.makedirs(out_dir, exist_ok=True)

    # --- Read preprocessed chartevents ONLY ---
    print("Reading chartevents (PREPROCESSED only)...")
    ce_dir = os.path.join(data_dir, 'intermediates')
    ce_paths = [p for p in os.listdir(ce_dir) if p.startswith("ce") and p.endswith(".csv")]
    if len(ce_paths) == 0:
        raise FileNotFoundError(f"No preprocessed ce*.csv found in {ce_dir}. Run preprocess first.")

    chart_events = ChartEvents([
        pd.read_csv(os.path.join(ce_dir, p), dtype={C_ITEMID: int})
        for p in ce_paths
    ])

    print("Reading onset data...")
    onset_data = load_csv(os.path.join(data_dir, 'intermediates', 'sepsis_onset.csv'))

    print("Reading demog...")
    demog = load_csv(os.path.join(data_dir, 'intermediates', 'demog.csv'))

    print("Reading labs (PREPROCESSED only)...")
    labs_ce_path = os.path.join(data_dir, 'intermediates', 'labs_ce.csv')
    labs_le_path = os.path.join(data_dir, 'intermediates', 'labs_le.csv')
    labs_ce = pd.read_csv(labs_ce_path, dtype={C_ITEMID: int})
    labs_le = pd.read_csv(labs_le_path, dtype={C_ITEMID: int})
    labU = pd.concat([labs_ce, labs_le], ignore_index=True)

    print("Reading mechvent_pe (from intermediates if present)...")
    mv_path = os.path.join(data_dir, 'intermediates', 'mechvent_pe.csv')
    if os.path.exists(mv_path):
        MV_procedure = pd.read_csv(mv_path)
        # ensure expected col names exist
        MV_procedure = MV_procedure[~pd.isna(MV_procedure[C_ICUSTAYID])]
    else:
        MV_procedure = None

    if args.filter_stays_path:
        print("Reading filter stays...")
        allowed_stays_df = load_csv(args.filter_stays_path)
        allowed_stays = allowed_stays_df[C_ICUSTAYID]
        old_count = len(onset_data)
        onset_data = onset_data[onset_data[C_ICUSTAYID].isin(allowed_stays)]
        print(f"Filtered from {old_count} to {len(onset_data)} ICU stay ids")

    if args.head:
        onset_data = onset_data.head(args.head)

    state_df, qstime = build_patient_states(
        chart_events=chart_events,
        onset_data=onset_data,
        demog=demog,
        labU=labU,
        MV_procedure=MV_procedure,
        winb4=args.window_before,
        winaft=args.window_after
    )

    print(f"Result: state_df contains {len(state_df)} rows, {len(state_df.columns)} columns")

    state_df.to_csv(os.path.join(out_dir, 'patient_states.csv'), index=False, float_format='%g')
    qstime.to_csv(os.path.join(out_dir, 'qstime.csv'), float_format='%g')