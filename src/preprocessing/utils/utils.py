import pickle

import pandas as pd
from typing import Any, Tuple,  cast
import os
from preprocessing.utils.columns import *
from typing import List, Optional

def load_csv(*file_paths: str, null_icustayid: bool = False, **kwargs: Any) -> pd.DataFrame:
    """
    Attempts to load a data CSV from the file paths given, and returns the first
    one whose file path exists.
    """
    for path in file_paths:
        if os.path.exists(path):
            spec = STAY_ID_OPTIONAL_DTYPE_SPEC if null_icustayid else DTYPE_SPEC
            return pd.read_csv(path, dtype=cast(Any, spec), **kwargs)
    raise FileNotFoundError(", ".join(file_paths))

def load_intermediate_or_raw_csv(data_dir: str, file_name: str) -> pd.DataFrame:
    return load_csv(
        os.path.join(data_dir, "intermediates", file_name),
        os.path.join(data_dir, "raw_data", file_name),
    )

def reverse_readline(filename, buf_size=8192):
    """A generator that returns the lines of a file in reverse order"""
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # The first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # If the previous chunk starts right from the beginning of line
                # do not concat the segment to the last line of new chunk.
                # Instead, yield the segment first 
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment

def save_pickle(path: str, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_is_first(length: int) -> np.ndarray:
    arr = np.zeros(length, dtype=np.float32)
    if length > 0:
        arr[0] = 1.0
    return arr


def build_is_terminal(length: int) -> np.ndarray:
    arr = np.zeros(length, dtype=np.float32)
    if length > 0:
        arr[-1] = 1.0
    return arr


def build_discount_sequence(length: int) -> np.ndarray:
    arr = np.ones(length, dtype=np.float32)
    if length > 0:
        arr[-1] = 0.0
    return arr


def parse_column_list(arg_value: Optional[str]) -> Optional[List[str]]:
    if arg_value is None:
        return None
    cols = [x.strip() for x in arg_value.split(",") if x.strip()]
    return cols if cols else None


def validate_columns_exist(df: pd.DataFrame, cols: List[str], group_name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {group_name} columns: {missing}")


def infer_default_action_columns(df: pd.DataFrame) -> List[str]:
    candidates = [C_INPUT_STEP, C_MAX_DOSE_VASO]
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        raise ValueError(
            "No default action columns found in actions file. "
            "Pass --action-cols explicitly."
        )
    return cols


def check_unique_keys(df: pd.DataFrame, df_name: str):
    required = [C_ICUSTAYID, C_TIMESTEP]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required key columns: {missing}")

    dup_count = df.duplicated(subset=[C_ICUSTAYID, C_TIMESTEP]).sum()
    if dup_count > 0:
        raise ValueError(
            f"{df_name} has {int(dup_count)} duplicated rows on "
            f"({C_ICUSTAYID}, {C_TIMESTEP})"
        )


def sort_by_keys(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values([C_ICUSTAYID, C_TIMESTEP]).reset_index(drop=True)


def get_mask_feature_cols(mask_df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    mask_cols = [c for c in feature_cols if c in mask_df.columns]
    missing = [c for c in feature_cols if c not in mask_df.columns]
    if missing:
        raise ValueError(
            f"mask file is missing feature columns: {missing}"
        )
    return mask_cols


def get_delta_feature_cols(delta_df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    delta_cols = [c for c in feature_cols if c in delta_df.columns]
    missing = [c for c in feature_cols if c not in delta_df.columns]
    if missing:
        raise ValueError(
            f"delta file is missing feature columns: {missing}"
        )
    return delta_cols


def filter_by_stays(df: pd.DataFrame, stay_ids: np.ndarray) -> pd.DataFrame:
    return df[df[C_ICUSTAYID].isin(stay_ids)].copy()

def fit_action_bins(input_amounts, vaso_doses, n_action_bins=5):
    input_amounts = np.clip(np.asarray(input_amounts, dtype=np.float32), 0.0, None)
    vaso_doses = np.clip(np.asarray(vaso_doses, dtype=np.float32), 0.0, None)

    percentiles = np.linspace(0, 100, n_action_bins)[1:-1]  # [25, 50, 75]

    input_cutoffs = np.percentile(input_amounts[input_amounts > 0], percentiles).tolist()
    vaso_cutoffs = np.percentile(vaso_doses[vaso_doses > 0], percentiles).tolist()

    io = np.zeros_like(input_amounts, dtype=np.int64)
    vc = np.zeros_like(vaso_doses, dtype=np.int64)

    io[input_amounts > 0] = np.digitize(input_amounts[input_amounts > 0], input_cutoffs) + 1
    vc[vaso_doses > 0] = np.digitize(vaso_doses[vaso_doses > 0], vaso_cutoffs) + 1
    
    actions = io * n_action_bins + vc

    median_inputs = [
        float(np.median(input_amounts[io == b])) if np.any(io == b) else 0.0
        for b in range(n_action_bins)
    ]

    median_vaso = [
        float(np.median(vaso_doses[vc == b])) if np.any(vc == b) else 0.0
        for b in range(n_action_bins)
    ]

    return actions.astype(np.int64), (median_inputs, median_vaso), (input_cutoffs, vaso_cutoffs)

def transform_actions(input_amounts, vaso_doses, cutoffs):
    input_cutoffs, vaso_cutoffs = cutoffs
    n_action_bins = len(input_cutoffs) + 2  # 3 cutoff => 5 bin

    input_amounts = np.clip(np.asarray(input_amounts, dtype=np.float32), 0.0, None)
    vaso_doses = np.clip(np.asarray(vaso_doses, dtype=np.float32), 0.0, None)

    io = np.zeros_like(input_amounts, dtype=np.int64)
    vc = np.zeros_like(vaso_doses, dtype=np.int64)

    io[input_amounts > 0] = np.digitize(input_amounts[input_amounts > 0], input_cutoffs) + 1
    vc[vaso_doses > 0] = np.digitize(vaso_doses[vaso_doses > 0], vaso_cutoffs) + 1
    action_ids = io * n_action_bins + vc

    return action_ids.astype(np.int64)


def transform_actions_separate(input_amounts, vaso_doses, cutoffs):
    input_cutoffs, vaso_cutoffs = cutoffs
    n_action_bins = len(input_cutoffs) + 2

    input_amounts = np.clip(np.asarray(input_amounts, dtype=np.float32), 0.0, None)
    vaso_doses = np.clip(np.asarray(vaso_doses, dtype=np.float32), 0.0, None)

    io = np.zeros_like(input_amounts, dtype=np.int64)
    vc = np.zeros_like(vaso_doses, dtype=np.int64)

    io[input_amounts > 0] = (
        np.digitize(input_amounts[input_amounts > 0], input_cutoffs) + 1
    )
    vc[vaso_doses > 0] = (
        np.digitize(vaso_doses[vaso_doses > 0], vaso_cutoffs) + 1
    )

    return io.astype(np.int64), vc.astype(np.int64)


def one_hot_actions(action_ids: np.ndarray, num_actions: int) -> np.ndarray:
    action_ids = np.asarray(action_ids, dtype=np.int64)
    out = np.zeros((len(action_ids), num_actions), dtype=np.float32)
    out[np.arange(len(action_ids)), action_ids] = 1.0
    return out


def save_npz(path: str, data: dict):
    np.savez_compressed(path, **data)