import os
import argparse
import pandas as pd
from tqdm import tqdm

from src.preprocessing.utils import load_csv
from src.preprocessing.columns import *

tqdm.pandas()

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

DEFAULT_FILES = [
    "abx.csv",
    "demog.csv",
    "culture.csv",
    "microbio.csv",
    "comorbidities.csv",

    # onset (from derived.sepsis3)
    "onset_derived.csv",

    # derived timeseries / intervals
    "weight_derived.csv",
    "vaso_derived.csv",
    "gcs_derived.csv",
    "vital_derived.csv",
    "bg_derived.csv",
    "fio2_derived.csv",
    "cbc_derived.csv",
    "labs_derived.csv",
    "ion_cal_derived.csv",
    "mag_derived.csv",
    "liver_derived.csv",
    "coag_derived.csv",
    "urine_derived.csv",
    "mechvent_derived.csv",
    "sofa_derived.csv",
    "fluid_mv.csv",
    "preadm_fluid.csv",
    "preadm_uo.csv",
]


def _drop_null_stay_id(df: pd.DataFrame) -> pd.DataFrame:
    if C_ICUSTAYID not in df.columns:
        return df
    return df[~pd.isna(df[C_ICUSTAYID])].copy()


def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by (icustayid, time) if possible.
    Timeseries tables typically have charttime; interval tables have start/endtime.
    """
    cols = list(df.columns)

    if C_ICUSTAYID in cols:
        if C_CHARTTIME in cols:
            return df.sort_values([C_ICUSTAYID, C_CHARTTIME], kind="mergesort")
        if C_STARTTIME in cols:
            # if both start/end exist, sort by start then end
            if C_ENDTIME in cols:
                return df.sort_values([C_ICUSTAYID, C_STARTTIME, C_ENDTIME], kind="mergesort")
            return df.sort_values([C_ICUSTAYID, C_STARTTIME], kind="mergesort")

    # fallback
    return df


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    # Drop exact duplicate rows (safe + helps size)
    return df.drop_duplicates()


def _basic_casts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal dtype normalization:
    - icustayid -> Int64 or int64 depending on your conventions
    - times -> Int64 when they are unix seconds
    """
    if C_ICUSTAYID in df.columns:
        # keep nullable if any
        try:
            df[C_ICUSTAYID] = df[C_ICUSTAYID].astype("Int64")
        except Exception:
            pass

    for tcol in [C_CHARTTIME, C_STARTTIME, C_ENDTIME]:
        if tcol in df.columns:
            # these are UNIX_SECONDS in your queries -> integers (nullable)
            try:
                df[tcol] = df[tcol].astype("Int64")
            except Exception:
                pass

    return df


def preprocess_one(input_path: str, output_path: str) -> bool:
    """
    Returns True if file processed, False if missing.
    """
    if not os.path.exists(input_path):
        print(f"[skip] missing: {os.path.basename(input_path)}")
        return False

    # Note: null_icustayid=True avoids hard-failing on missing stay ids;
    # we explicitly drop them for MedDreamer.
    df = load_csv(input_path, null_icustayid=True)

    # MedDreamer: require a valid stay_id for alignment
    df = _drop_null_stay_id(df)

    # demog cleanup (keep as before, but no imputations)
    if os.path.basename(input_path) == "demog.csv":
        for col, fill in [(C_MORTA_90, 0), (C_MORTA_HOSP, 0), (C_ELIXHAUSER, 0)]:
            if col in df.columns:
                df.loc[pd.isna(df[col]), col] = fill

    df = _basic_casts(df)
    df = _dedup(df)
    df = _sort_df(df)

    df.to_csv(output_path, index=False)
    print(f"[ok] {os.path.basename(input_path)} -> {os.path.basename(output_path)} ({len(df)} rows)")
    return True


def build_bacterio_and_clean_abx(input_dir, output_dir):

    culture_path = os.path.join(input_dir, "culture.csv")
    microbio_path = os.path.join(input_dir, "microbio.csv")
    abx_path = os.path.join(input_dir, "abx.csv")

    if not (os.path.exists(culture_path) and os.path.exists(microbio_path)):
        print("[skip] bacterio build: missing culture or microbio")
        return

    print("Building bacterio.csv")

    culture = load_csv(culture_path, null_icustayid=True)
    microbio = load_csv(microbio_path, null_icustayid=True)

    # fill missing charttime with chartdate
    if C_CHARTTIME in microbio.columns and C_CHARTDATE in microbio.columns:
        ii = microbio[C_CHARTTIME].isnull()
        microbio.loc[ii, C_CHARTTIME] = microbio.loc[ii, C_CHARTDATE]

    cols = [C_SUBJECT_ID, C_HADM_ID, C_ICUSTAYID, C_CHARTTIME]

    bacterio = pd.concat([
        microbio[cols],
        culture[cols]
    ], ignore_index=True)

    bacterio = bacterio.dropna(subset=[C_ICUSTAYID])
    bacterio = bacterio.drop_duplicates()

    bacterio.to_csv(os.path.join(output_dir, "bacterio.csv"), index=False)

    print(f"[ok] bacterio.csv ({len(bacterio)} rows)")

    # Clean ABX
    if not os.path.exists(abx_path):
        print("[skip] abx clean: missing abx file")
        return

    print("Cleaning abx.csv")

    abx = load_csv(abx_path, null_icustayid=True)

    abx = abx[
        (~pd.isna(abx[C_STARTDATE])) &
        (~pd.isna(abx[C_ICUSTAYID]))
    ]

    abx = abx.drop_duplicates()

    abx.to_csv(os.path.join(output_dir, "abx.csv"), index=False)

    print(f"[ok] abx.csv ({len(abx)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "MedDreamer preprocessing: minimal cleaning of extracted CSVs "
            "(derived tables + onset + static). No binning, no itemid remap, "
        )
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        type=str,
        default=None,
        help="Directory to read files from (default: data/raw_data)",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to write files to (default: data/intermediates)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional explicit list of CSV filenames to process (overrides default list).",
    )

    args = parser.parse_args()

    in_dir = args.input_dir or os.path.join(PARENT_DIR, "data", "raw_data")
    out_dir = args.output_dir or os.path.join(PARENT_DIR, "data", "intermediates")
    os.makedirs(out_dir, exist_ok=True)

    build_bacterio_and_clean_abx(in_dir, out_dir)

    file_list = args.files if args.files is not None and len(args.files) > 0 else DEFAULT_FILES

    print("Input dir :", in_dir)
    print("Output dir:", out_dir)
    print("Files     :", ", ".join(file_list))

    processed = 0
    for fname in tqdm(file_list):
        inp = os.path.join(in_dir, fname)
        out = os.path.join(out_dir, fname)
        if preprocess_one(inp, out):
            processed += 1

    print(f"Done. Processed {processed}/{len(file_list)} files.")