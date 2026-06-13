import pandas as pd
import os
import argparse
from tqdm import tqdm

from preprocessing.utils.columns import *
from preprocessing.utils.utils import load_csv

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


def _first_timestep_sources(first_t, stay_point, stay_interval):
    """
    Return the list of table sources that contribute the given timestep.
    For point tables: exact charttime match.
    For interval tables: exact starttime match.
    """
    sources = []

    for name, s in stay_point.items():
        if s is not None and C_CHARTTIME in s.columns:
            if (s[C_CHARTTIME] == first_t).any():
                sources.append(f"point:{name}")

    for name, s in stay_interval.items():
        if s is not None and C_STARTTIME in s.columns:
            if (s[C_STARTTIME] == first_t).any():
                sources.append(f"interval:{name}")

    return sources


def _is_interval_active(df, t):
    """
    Check whether an interval table has at least one active row at time t.
    """
    if df is None or len(df) == 0:
        return False
    return ((df[C_STARTTIME] <= t) & (t <= df[C_ENDTIME])).any()


def _get_active_interval_rows(df, t):
    """
    Return active rows of an interval table at time t.
    """
    if df is None or len(df) == 0:
        return None
    active = df[(df[C_STARTTIME] <= t) & (t <= df[C_ENDTIME])]
    if len(active) == 0:
        return None
    return active


def _has_point_observation_at_t(stay_point, t):
    """
    True if any point-event table has a measurement exactly at t.
    """
    for _, s in stay_point.items():
        if s is not None and C_CHARTTIME in s.columns:
            if (s[C_CHARTTIME] == t).any():
                return True
    return False


def _point_tables_at_t(stay_point, t):
    """
    Return point table names contributing exactly at t.
    """
    hits = []
    for name, s in stay_point.items():
        if s is not None and C_CHARTTIME in s.columns:
            if (s[C_CHARTTIME] == t).any():
                hits.append(name)
    return hits


def _mechvent_observed_at_t(mechvent_df, t):
    """
    Decide whether mechvent is a valid clinical trigger at timestep t.

    We allow mechvent to start an episode if it is clinically available at t:
    - active at t
    - and its status is not null / not 'none'
    """
    active = _get_active_interval_rows(mechvent_df, t)
    if active is None:
        return False

    row = active.iloc[-1]
    status = row.get("mechvent", row.get("ventilation_status", None))

    if pd.isna(status) or status is None:
        return False

    return int(str(status).lower() != "none") == 1


def _is_valid_episode_start_timestep(t, stay_point, stay_interval):
    """
    Episode can start at t if:
    - any real point observation exists at t
    - OR mechvent is clinically observed/active at t

    Episode cannot start from:
    - weight alone
    - sofa alone
    """
    if _has_point_observation_at_t(stay_point, t):
        return True

    if _mechvent_observed_at_t(stay_interval.get("mechvent"), t):
        return True

    return False


def _find_first_valid_timestep(timesteps, stay_point, stay_interval):
    """
    Return the first timestep that is allowed to start the episode.
    """
    for t in timesteps:
        if _is_valid_episode_start_timestep(t, stay_point, stay_interval):
            return t
    return None


def build_patient_states_derived(
    onset_data,
    demog,
    vital, bg, fio2, cbc, labs, ionca, mag, liver, coag, gcs, urine,
    mechvent, weight, sofa,
    winb4, winaft,
    debug_first_timestep=False,
    debug_examples=20,
):
    combined_data = []
    infection_times = []
    first_timestep_debug = []

    buffer = 2
    bounds = ((winb4 + buffer) * 3600, (winaft + buffer) * 3600)

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

    interval_tables = [
        ("mechvent", mechvent, C_STARTTIME, C_ENDTIME),
        ("weight",   weight,   C_STARTTIME, C_ENDTIME),
        ("sofa",     sofa,     C_STARTTIME, C_ENDTIME),
    ]

    printed_examples = 0

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

        # Fetch and window point tables around onset.
        stay_point = {}
        for name, df, tcol in point_tables:
            s = _fetch_stay(df, icustayid)
            if s is None:
                stay_point[name] = None
                continue
            s = time_window(s, tcol, qst, *bounds)
            stay_point[name] = s if (s is not None and len(s) > 0) else None

        # Fetch and window interval tables by interval overlap.
        stay_interval = {}
        lower = qst - bounds[0]
        upper = qst + bounds[1]

        for name, df, scol, ecol in interval_tables:
            s = _fetch_stay(df, icustayid)
            if s is None:
                stay_interval[name] = None
                continue

            s = s[(s[ecol] >= lower) & (s[scol] <= upper)]
            stay_interval[name] = s if (len(s) > 0) else None

        # Timeline = all point charttimes + all interval starttimes
        ts_series = []
        for name, _, tcol in point_tables:
            s = stay_point.get(name)
            if s is not None and tcol in s.columns:
                ts_series.append(s[tcol])

        for name, _, scol, _ in interval_tables:
            if name not in ("mechvent",):
                continue

            s = stay_interval.get(name)
            if s is not None and scol in s.columns:
                ts_series.append(s[scol])

        timesteps = _union_timesteps(*ts_series)
        if len(timesteps) == 0:
            continue

        raw_first_t = timesteps[0]
        valid_first_t = _find_first_valid_timestep(timesteps, stay_point, stay_interval)

        # If we never get a real observation or clinically meaningful mechvent,
        # skip the stay entirely.
        if valid_first_t is None:
            first_timestep_debug.append({
                C_ICUSTAYID: icustayid,
                C_ONSET_TIME: qst,
                "raw_first_timestep": raw_first_t,
                "valid_first_timestep": pd.NA,
                "delta_hours_raw_first_t_minus_onset": (raw_first_t - qst) / 3600.0,
                "delta_hours_valid_first_t_minus_onset": pd.NA,
                "raw_sources": "|".join(_first_timestep_sources(raw_first_t, stay_point, stay_interval)) or "unknown",
                "valid_sources": "",
                "has_sofa_active_at_raw_first_t": int(_is_interval_active(stay_interval.get("sofa"), raw_first_t)),
                "has_weight_active_at_raw_first_t": int(_is_interval_active(stay_interval.get("weight"), raw_first_t)),
                "has_mechvent_active_at_raw_first_t": int(_is_interval_active(stay_interval.get("mechvent"), raw_first_t)),
                "point_tables_at_raw_first_t": "|".join(_point_tables_at_t(stay_point, raw_first_t)),
                "raw_first_t_is_valid_episode_start": 0,
                "stay_skipped_no_valid_start": 1,
            })
            continue

        raw_first_sources = _first_timestep_sources(raw_first_t, stay_point, stay_interval)
        valid_first_sources = _first_timestep_sources(valid_first_t, stay_point, stay_interval)

        sofa_active_at_raw_first_t = _is_interval_active(stay_interval.get("sofa"), raw_first_t)
        weight_active_at_raw_first_t = _is_interval_active(stay_interval.get("weight"), raw_first_t)
        mechvent_active_at_raw_first_t = _is_interval_active(stay_interval.get("mechvent"), raw_first_t)

        sofa_active_at_valid_first_t = _is_interval_active(stay_interval.get("sofa"), valid_first_t)
        weight_active_at_valid_first_t = _is_interval_active(stay_interval.get("weight"), valid_first_t)
        mechvent_active_at_valid_first_t = _is_interval_active(stay_interval.get("mechvent"), valid_first_t)

        point_hits_at_raw_first_t = _point_tables_at_t(stay_point, raw_first_t)
        point_hits_at_valid_first_t = _point_tables_at_t(stay_point, valid_first_t)

        first_timestep_debug.append({
            C_ICUSTAYID: icustayid,
            C_ONSET_TIME: qst,
            "raw_first_timestep": raw_first_t,
            "valid_first_timestep": valid_first_t,
            "delta_hours_raw_first_t_minus_onset": (raw_first_t - qst) / 3600.0,
            "delta_hours_valid_first_t_minus_onset": (valid_first_t - qst) / 3600.0,
            "raw_sources": "|".join(raw_first_sources) if len(raw_first_sources) > 0 else "unknown",
            "valid_sources": "|".join(valid_first_sources) if len(valid_first_sources) > 0 else "unknown",
            "has_sofa_active_at_raw_first_t": int(sofa_active_at_raw_first_t),
            "has_weight_active_at_raw_first_t": int(weight_active_at_raw_first_t),
            "has_mechvent_active_at_raw_first_t": int(mechvent_active_at_raw_first_t),
            "has_sofa_active_at_valid_first_t": int(sofa_active_at_valid_first_t),
            "has_weight_active_at_valid_first_t": int(weight_active_at_valid_first_t),
            "has_mechvent_active_at_valid_first_t": int(mechvent_active_at_valid_first_t),
            "point_tables_at_raw_first_t": "|".join(point_hits_at_raw_first_t) if len(point_hits_at_raw_first_t) > 0 else "",
            "point_tables_at_valid_first_t": "|".join(point_hits_at_valid_first_t) if len(point_hits_at_valid_first_t) > 0 else "",
            "raw_first_t_is_valid_episode_start": int(raw_first_t == valid_first_t),
            "stay_skipped_no_valid_start": 0,
        })

        if debug_first_timestep:
            raw_invalid_because_interval_only = (
                raw_first_t != valid_first_t and
                len(raw_first_sources) > 0 and
                all(src.startswith("interval:") for src in raw_first_sources)
            )

            if raw_invalid_because_interval_only and printed_examples < debug_examples:
                print(
                    f"[DEBUG shifted episode start] stay={icustayid} "
                    f"raw_first_t={raw_first_t} valid_first_t={valid_first_t} onset={qst} "
                    f"raw_delta_h={(raw_first_t - qst) / 3600.0:.3f} "
                    f"valid_delta_h={(valid_first_t - qst) / 3600.0:.3f} "
                    f"raw_sources={raw_first_sources} valid_sources={valid_first_sources}",
                    flush=True,
                )
                printed_examples += 1

        # Keep only timesteps from the first valid episode-start onward.
        timesteps = [t for t in timesteps if t >= valid_first_t]
        if len(timesteps) == 0:
            continue

        # Build one row per final timestep (collected per-patient for UO post-processing).
        patient_items = []
        for i, t in enumerate(timesteps):
            item = {
                C_BLOC: i,
                C_ICUSTAYID: icustayid,
                C_TIMESTEP: t,
            }

            # Static features copied to every row after episode start.
            item[C_AGE] = age
            item[C_GENDER] = gender
            item[C_ELIXHAUSER] = elix
            item[C_RE_ADMISSION] = readmission

            # Copy point-event values at exact charttime.
            for name, _, tcol in point_tables:
                s = stay_point.get(name)
                if s is None:
                    continue

                rows = s[s[tcol] == t]
                if len(rows) == 0:
                    continue

                r = rows.sort_index().iloc[-1]
                for col in rows.columns:
                    if col in (C_ICUSTAYID, tcol):
                        continue
                    item[col] = r[col]

            # Mechvent interval.
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

            # Weight interval.
            wt = stay_interval.get("weight")
            if wt is not None and len(wt) > 0:
                active = wt[(wt[C_STARTTIME] <= t) & (t <= wt[C_ENDTIME])]
                if len(active) > 0:
                    item[C_WEIGHT] = active.iloc[-1].get(C_WEIGHT, active.iloc[-1].get("Weight_kg", pd.NA))

            # SOFA interval.
            sf = stay_interval.get("sofa")
            if sf is not None and len(sf) > 0:
                active = sf[(sf[C_STARTTIME] <= t) & (t <= sf[C_ENDTIME])]
                if len(active) > 0:
                    item[C_SOFA] = active.iloc[-1].get(C_SOFA, active.iloc[-1].get("sofa", pd.NA))

            patient_items.append(item)

        # Convert urine_output from ml/event to ml/h rate.
        # Raw values depend on measurement interval (1h event → ~50ml, 6h event → ~300ml
        # for the same true production rate). Dividing by delta_t makes the feature
        # a true instantaneous rate comparable across irregular measurement schedules.
        _uo_indices = [
            j for j, it in enumerate(patient_items)
            if C_URINE_OUTPUT in it and it[C_URINE_OUTPUT] is not pd.NA
            and not (isinstance(it[C_URINE_OUTPUT], float) and pd.isna(it[C_URINE_OUTPUT]))
        ]
        for j_pos, j in enumerate(_uo_indices):
            if j_pos == 0:
                # No previous measurement within this episode — set to missing.
                patient_items[j][C_URINE_OUTPUT] = pd.NA
            else:
                prev_j = _uo_indices[j_pos - 1]
                delta_s = float(patient_items[j][C_TIMESTEP]) - float(patient_items[prev_j][C_TIMESTEP])
                delta_h = delta_s / 3600.0
                if delta_h > 0:
                    rate = float(patient_items[j][C_URINE_OUTPUT]) / delta_h
                    # Cap at 500 ml/h to suppress data-entry outliers.
                    patient_items[j][C_URINE_OUTPUT] = min(rate, 500.0)
                else:
                    patient_items[j][C_URINE_OUTPUT] = pd.NA

        combined_data.extend(patient_items)

        infection_times.append({
            C_ICUSTAYID: icustayid,
            C_ONSET_TIME: qst,
            C_FIRST_TIMESTEP: timesteps[0],
            C_LAST_TIMESTEP: timesteps[-1],
            C_DISCHTIME: dischtime,
        })

    state_df = pd.DataFrame(combined_data)
    qstime = pd.DataFrame(infection_times).set_index(C_ICUSTAYID)

    debug_df = pd.DataFrame(first_timestep_debug)

    if len(debug_df) > 0:
        print("\n=== RAW FIRST TIMESTEP SOURCE SUMMARY ===")
        print(debug_df["raw_sources"].value_counts().head(20))

        print("\n=== VALID FIRST TIMESTEP SOURCE SUMMARY ===")
        valid_nonempty = debug_df.loc[
            debug_df["stay_skipped_no_valid_start"] == 0,
            "valid_sources"
        ]
        print(valid_nonempty.value_counts().head(20))

        print("\n=== RAW FIRST TIMESTEPS THAT WERE SHIFTED ===")
        shifted = debug_df.loc[debug_df["raw_first_t_is_valid_episode_start"] == 0, "raw_sources"]
        print(shifted.value_counts().head(20))

        print("\n=== STAYS SKIPPED (NO VALID EPISODE START) ===")
        print(int(debug_df["stay_skipped_no_valid_start"].sum()))

    return state_df, qstime, debug_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        "Build patient states from MIMIC-IV derived tables (MedDreamer style), "
        "starting each episode at the first valid clinical observation "
        "(point observation or clinically meaningful mechvent)."
    ))
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory in which to output (e.g. data/intermediates/patient_states)"
    )
    parser.add_argument(
        "--data",
        dest="data_dir",
        type=str,
        default=None,
        help="Data directory (default: ../data)"
    )
    parser.add_argument(
        "--window-before",
        dest="window_before",
        type=int,
        default=24
    )
    parser.add_argument(
        "--window-after",
        dest="window_after",
        type=int,
        default=48
    )
    parser.add_argument(
        "--head",
        dest="head",
        type=int,
        default=None
    )
    parser.add_argument(
        "--debug-first-timestep",
        action="store_true",
        help="Print examples where the raw first timestep is shifted to a later valid episode start."
    )
    parser.add_argument(
        "--debug-examples",
        type=int,
        default=20,
        help="Maximum number of shifted-start debug examples to print."
    )
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(PARENT_DIR, "data")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    interm = os.path.join(data_dir, "intermediates")

    print("Reading onset (derived)...")
    onset = load_csv(os.path.join(interm, "onset_derived.csv"))

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
    vital = r("vital_derived")
    bg = r("bg_derived")
    fio2 = r("fio2_derived")
    cbc = r("cbc_derived")
    labs = r("labs_derived")
    ionca = r("ion_cal_derived")
    mag = r("mag_derived")
    liver = r("liver_derived")
    coag = r("coag_derived")
    gcs = r("gcs_derived")
    urine = r("urine_derived")
    mechvent = r("mechvent_derived")
    weight = r("weight_derived")
    sofa = r("sofa_derived")

    state_df, qstime, debug_df = build_patient_states_derived(
        onset_data=onset,
        demog=demog,
        vital=vital,
        bg=bg,
        fio2=fio2,
        cbc=cbc,
        labs=labs,
        ionca=ionca,
        mag=mag,
        liver=liver,
        coag=coag,
        gcs=gcs,
        urine=urine,
        mechvent=mechvent,
        weight=weight,
        sofa=sofa,
        winb4=args.window_before,
        winaft=args.window_after,
        debug_first_timestep=args.debug_first_timestep,
        debug_examples=args.debug_examples,
    )

    print(f"Result: state_df={len(state_df)} rows, {len(state_df.columns)} cols")

    state_df.to_csv(
        os.path.join(out_dir, "patient_states.csv"),
        index=False,
        float_format="%g"
    )

    qstime.to_csv(
        os.path.join(out_dir, "qstime.csv"),
        float_format="%g"
    )

    debug_df.to_csv(
        os.path.join(out_dir, "first_timestep_debug.csv"),
        index=False
    )

    print(f"Saved debug file to: {os.path.join(out_dir, 'first_timestep_debug.csv')}")