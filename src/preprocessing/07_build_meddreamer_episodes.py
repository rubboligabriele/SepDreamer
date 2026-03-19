import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.utils import (
    load_csv,
    save_pickle,
    ensure_dir,
    filter_by_stays,
    sort_by_keys,
    check_unique_keys,
    validate_columns_exist,
    parse_column_list,
    infer_default_action_columns,
    get_mask_feature_cols,
    get_delta_feature_cols,
    build_is_first,
    build_is_terminal,
    build_discount_sequence,
)
from src.preprocessing.columns import *
from src.preprocessing.normalization import DataNormalization


def build_indexed_pair_lookup(
    states_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> Tuple[set, set, set, set]:
    states_keys = set(zip(states_df[C_ICUSTAYID], states_df[C_TIMESTEP]))
    actions_keys = set(zip(actions_df[C_ICUSTAYID], actions_df[C_TIMESTEP]))
    mask_keys = set(zip(mask_df[C_ICUSTAYID], mask_df[C_TIMESTEP]))
    delta_keys = set(zip(delta_df[C_ICUSTAYID], delta_df[C_TIMESTEP]))
    return states_keys, actions_keys, mask_keys, delta_keys


def check_global_alignment(
    states_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    delta_df: pd.DataFrame,
):
    states_keys, actions_keys, mask_keys, delta_keys = build_indexed_pair_lookup(
        states_df, actions_df, mask_df, delta_df
    )

    if states_keys != actions_keys:
        only_states = len(states_keys - actions_keys)
        only_actions = len(actions_keys - states_keys)
        raise ValueError(
            f"states/actions key mismatch on ({C_ICUSTAYID}, {C_TIMESTEP}). "
            f"Only in states: {only_states}, only in actions: {only_actions}"
        )

    if states_keys != mask_keys:
        only_states = len(states_keys - mask_keys)
        only_mask = len(mask_keys - states_keys)
        raise ValueError(
            f"states/mask key mismatch on ({C_ICUSTAYID}, {C_TIMESTEP}). "
            f"Only in states: {only_states}, only in mask: {only_mask}"
        )

    if states_keys != delta_keys:
        only_states = len(states_keys - delta_keys)
        only_delta = len(delta_keys - states_keys)
        raise ValueError(
            f"states/delta key mismatch on ({C_ICUSTAYID}, {C_TIMESTEP}). "
            f"Only in states: {only_states}, only in delta: {only_delta}"
        )


def prepare_sequences_for_stays(
    states_raw: pd.DataFrame,
    states_norm: pd.DataFrame,
    actions_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    stay_ids: np.ndarray,
    feature_cols: List[str],
    action_cols: List[str],
    mask_cols: List[str],
    delta_cols: List[str],
    reward_col: str,
    min_seq_len: int = 2,
    add_discount: bool = True,
) -> List[Dict[str, np.ndarray]]:
    sequences = []

    for stay_id in stay_ids:
        stay_states_raw = states_raw[states_raw[C_ICUSTAYID] == stay_id].copy()
        stay_states_norm = states_norm[states_norm[C_ICUSTAYID] == stay_id].copy()
        stay_actions = actions_df[actions_df[C_ICUSTAYID] == stay_id].copy()
        stay_mask = mask_df[mask_df[C_ICUSTAYID] == stay_id].copy()
        stay_delta = delta_df[delta_df[C_ICUSTAYID] == stay_id].copy()

        if len(stay_states_raw) == 0:
            continue

        stay_states_raw = stay_states_raw.sort_values(C_TIMESTEP).reset_index(drop=True)
        stay_states_norm = stay_states_norm.sort_values(C_TIMESTEP).reset_index(drop=True)
        stay_actions = stay_actions.sort_values(C_TIMESTEP).reset_index(drop=True)
        stay_mask = stay_mask.sort_values(C_TIMESTEP).reset_index(drop=True)
        stay_delta = stay_delta.sort_values(C_TIMESTEP).reset_index(drop=True)

        if len(stay_states_raw) < min_seq_len:
            continue

        t_ref = stay_states_raw[C_TIMESTEP].values

        if not np.array_equal(t_ref, stay_states_norm[C_TIMESTEP].values):
            raise ValueError(f"Timestep mismatch between raw/norm states for stay {stay_id}")
        if not np.array_equal(t_ref, stay_actions[C_TIMESTEP].values):
            raise ValueError(f"Timestep mismatch between states/actions for stay {stay_id}")
        if not np.array_equal(t_ref, stay_mask[C_TIMESTEP].values):
            raise ValueError(f"Timestep mismatch between states/mask for stay {stay_id}")
        if not np.array_equal(t_ref, stay_delta[C_TIMESTEP].values):
            raise ValueError(f"Timestep mismatch between states/delta for stay {stay_id}")

        timesteps = stay_states_raw[C_TIMESTEP].astype(np.float32).values
        features = stay_states_norm[feature_cols].astype(np.float32).values
        actions = stay_actions[action_cols].astype(np.float32).fillna(0.0).values
        mask = stay_mask[mask_cols].astype(np.float32).fillna(0.0).values
        delta = stay_delta[delta_cols].astype(np.float32).fillna(0.0).values

        if reward_col not in stay_states_raw.columns:
            raise ValueError(f"Reward column '{reward_col}' not found in states file")
        reward = stay_states_raw[reward_col].astype(np.float32).fillna(0.0).values

        is_first = build_is_first(len(stay_states_raw))
        is_terminal = build_is_terminal(len(stay_states_raw))

        traj = {
            "icustayid": np.array([stay_id], dtype=np.int64),
            "timestep": timesteps,
            "features": features,
            "action": actions,
            "reward": reward,
            "mask": mask,
            "delta": delta,
            "is_first": is_first,
            "is_terminal": is_terminal,
        }

        if add_discount:
            traj["discount"] = build_discount_sequence(len(stay_states_raw)).astype(np.float32)

        sequences.append(traj)

    return sequences


def save_split_dataset(
    out_dir: str,
    split_name: str,
    states_raw: pd.DataFrame,
    states_norm: pd.DataFrame,
    actions_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    split_stay_ids: np.ndarray,
    feature_cols: List[str],
    action_cols: List[str],
    mask_cols: List[str],
    delta_cols: List[str],
    reward_col: str,
):
    states_raw_split = filter_by_stays(states_raw, split_stay_ids)
    states_norm_split = filter_by_stays(states_norm, split_stay_ids)
    actions_split = filter_by_stays(actions_df, split_stay_ids)
    mask_split = filter_by_stays(mask_df, split_stay_ids)
    delta_split = filter_by_stays(delta_df, split_stay_ids)

    sequences = prepare_sequences_for_stays(
        states_raw=states_raw_split,
        states_norm=states_norm_split,
        actions_df=actions_split,
        mask_df=mask_split,
        delta_df=delta_split,
        stay_ids=np.array(sorted(split_stay_ids)),
        feature_cols=feature_cols,
        action_cols=action_cols,
        mask_cols=mask_cols,
        delta_cols=delta_cols,
        reward_col=reward_col,
        min_seq_len=2,
        add_discount=True,
    )

    payload = {
        "trajectories": sequences,
        "metadata": {
            "split": split_name,
            "n_trajectories": len(sequences),
            "n_rows_states": len(states_raw_split),
            "n_rows_actions": len(actions_split),
            "n_rows_mask": len(mask_split),
            "n_rows_delta": len(delta_split),
            "n_stays": len(split_stay_ids),
            "feature_cols": feature_cols,
            "action_cols": action_cols,
            "mask_cols": mask_cols,
            "delta_cols": delta_cols,
            "reward_col": reward_col,
        },
    }

    out_path = os.path.join(out_dir, f"{split_name}.pkl")
    save_pickle(out_path, payload)

    print(
        f"[{split_name}] saved {len(sequences)} trajectories, "
        f"{len(split_stay_ids)} stays -> {out_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MedDreamer sequential dataset from separate CSV files"
    )
    parser.add_argument(
        "--states",
        type=str,
        required=True,
        help="Path to patient_states_with_reward.csv"
    )
    parser.add_argument(
        "--actions",
        type=str,
        required=True,
        help="Path to actions_filtered.csv"
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to mask_filtered.csv"
    )
    parser.add_argument(
        "--delta",
        type=str,
        required=True,
        help="Path to delta_filtered.csv"
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default=None,
        help="Optional path to sepsis_cohort.csv for stay filtering"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Train proportion at stay level"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation proportion at stay level (remaining goes to test)"
    )
    parser.add_argument(
        "--reward-col",
        type=str,
        default=C_REWARD,
        help="Reward column name in states file"
    )
    parser.add_argument(
        "--action-cols",
        type=str,
        default=None,
        help="Comma-separated action columns from actions file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    ensure_dir(args.output)

    print("Loading input files ...")
    states_df = load_csv(args.states)
    actions_df = load_csv(args.actions)
    mask_df = load_csv(args.mask)
    delta_df = load_csv(args.delta)

    if args.cohort is not None:
        cohort_df = load_csv(args.cohort)
        if C_ICUSTAYID not in cohort_df.columns:
            raise ValueError(f"Cohort file must contain column '{C_ICUSTAYID}'")
        keep_ids = cohort_df[C_ICUSTAYID].dropna().unique()
        states_df = filter_by_stays(states_df, keep_ids)
        actions_df = filter_by_stays(actions_df, keep_ids)
        mask_df = filter_by_stays(mask_df, keep_ids)
        delta_df = filter_by_stays(delta_df, keep_ids)
        print(f"Filtered by cohort: {len(keep_ids)} stays")

    check_unique_keys(states_df, "states")
    check_unique_keys(actions_df, "actions")
    check_unique_keys(mask_df, "mask")
    check_unique_keys(delta_df, "delta")

    states_df = sort_by_keys(states_df)
    actions_df = sort_by_keys(actions_df)
    mask_df = sort_by_keys(mask_df)
    delta_df = sort_by_keys(delta_df)

    check_global_alignment(states_df, actions_df, mask_df, delta_df)

    required_state_cols = [C_ICUSTAYID, C_TIMESTEP, args.reward_col]
    missing_required = [c for c in required_state_cols if c not in states_df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in states file: {missing_required}")

    feature_cols = [c for c in STATE_COLUMNS if c in states_df.columns]
    if not feature_cols:
        raise ValueError("No STATE_COLUMNS found in states file.")

    missing_state_cols = [c for c in STATE_COLUMNS if c not in states_df.columns]
    if missing_state_cols:
        print("Warning: these STATE_COLUMNS are missing and will be skipped:")
        for c in missing_state_cols:
            print(" -", c)

    action_cols = parse_column_list(args.action_cols)
    if action_cols is None:
        action_cols = infer_default_action_columns(actions_df)
    validate_columns_exist(actions_df, action_cols, "action")

    mask_cols = get_mask_feature_cols(mask_df, feature_cols)
    delta_cols = get_delta_feature_cols(delta_df, feature_cols)

    print("Feature columns:")
    print(feature_cols)
    print("Action columns:")
    print(action_cols)
    print("Mask columns:")
    print(mask_cols)
    print("Delta columns:")
    print(delta_cols)

    stay_ids = states_df[C_ICUSTAYID].dropna().unique()
    print(f"Total ICU stays: {len(stay_ids)}")

    train_ids, temp_ids = train_test_split(
        stay_ids,
        train_size=args.train_size,
        random_state=args.seed,
        shuffle=True,
    )

    val_ratio_over_temp = args.val_size / (1.0 - args.train_size)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_ratio_over_temp,
        random_state=args.seed,
        shuffle=True,
    )

    print(f"Train stays: {len(train_ids)}")
    print(f"Val stays:   {len(val_ids)}")
    print(f"Test stays:  {len(test_ids)}")

    train_states_df = states_df[states_df[C_ICUSTAYID].isin(train_ids)].copy()

    print("Fitting normalization on train states only ...")
    normer = DataNormalization(train_states_df)

    print("Applying normalization to all states ...")
    states_norm_features = normer.transform(states_df)

    non_feature_cols = [c for c in states_df.columns if c not in feature_cols]
    states_norm_df = pd.concat(
        [
            states_df[non_feature_cols].reset_index(drop=True),
            states_norm_features[feature_cols].reset_index(drop=True),
        ],
        axis=1,
    )

    normer_path = os.path.join(args.output, "normalization.pkl")
    if hasattr(normer, "save"):
        normer.save(normer_path)
    else:
        save_pickle(normer_path, normer)

    column_config = {
        "feature_cols": feature_cols,
        "action_cols": action_cols,
        "mask_cols": mask_cols,
        "delta_cols": delta_cols,
        "reward_col": args.reward_col,
        "state_columns_declared": STATE_COLUMNS,
        "all_feature_columns_declared": ALL_FEATURE_COLUMNS,
    }
    save_pickle(os.path.join(args.output, "column_config.pkl"), column_config)

    save_split_dataset(
        out_dir=args.output,
        split_name="train",
        states_raw=states_df,
        states_norm=states_norm_df,
        actions_df=actions_df,
        mask_df=mask_df,
        delta_df=delta_df,
        split_stay_ids=train_ids,
        feature_cols=feature_cols,
        action_cols=action_cols,
        mask_cols=mask_cols,
        delta_cols=delta_cols,
        reward_col=args.reward_col,
    )

    save_split_dataset(
        out_dir=args.output,
        split_name="val",
        states_raw=states_df,
        states_norm=states_norm_df,
        actions_df=actions_df,
        mask_df=mask_df,
        delta_df=delta_df,
        split_stay_ids=val_ids,
        feature_cols=feature_cols,
        action_cols=action_cols,
        mask_cols=mask_cols,
        delta_cols=delta_cols,
        reward_col=args.reward_col,
    )

    save_split_dataset(
        out_dir=args.output,
        split_name="test",
        states_raw=states_df,
        states_norm=states_norm_df,
        actions_df=actions_df,
        mask_df=mask_df,
        delta_df=delta_df,
        split_stay_ids=test_ids,
        feature_cols=feature_cols,
        action_cols=action_cols,
        mask_cols=mask_cols,
        delta_cols=delta_cols,
        reward_col=args.reward_col,
    )

    print("Done.")


if __name__ == "__main__":
    main()