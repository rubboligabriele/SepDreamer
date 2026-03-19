import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.preprocessing.utils import (
    load_csv,
    save_pickle,
    save_npz,
    ensure_dir,
    filter_by_stays,
    sort_by_keys,
    check_unique_keys,
    validate_columns_exist,
    infer_default_action_columns,
    get_mask_feature_cols,
    get_delta_feature_cols,
    build_is_first,
    build_is_terminal,
    build_discount_sequence,
    fit_action_bins,
    transform_actions,
    one_hot_actions,
)
from src.preprocessing.columns import *
from src.preprocessing.normalization import DataNormalization


def check_mask_delta_alignment(
    states_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    delta_df: pd.DataFrame,
):
    state_keys = set(zip(states_df[C_ICUSTAYID], states_df[C_TIMESTEP]))
    mask_keys = set(zip(mask_df[C_ICUSTAYID], mask_df[C_TIMESTEP]))
    delta_keys = set(zip(delta_df[C_ICUSTAYID], delta_df[C_TIMESTEP]))

    if state_keys != mask_keys:
        only_states = len(state_keys - mask_keys)
        only_mask = len(mask_keys - state_keys)
        raise ValueError(
            f"states/mask key mismatch on ({C_ICUSTAYID}, {C_TIMESTEP}). "
            f"Only in states: {only_states}, only in mask: {only_mask}"
        )

    if state_keys != delta_keys:
        only_states = len(state_keys - delta_keys)
        only_delta = len(delta_keys - state_keys)
        raise ValueError(
            f"states/delta key mismatch on ({C_ICUSTAYID}, {C_TIMESTEP}). "
            f"Only in states: {only_states}, only in delta: {only_delta}"
        )


def prepare_episode_for_stay(
    stay_id: int,
    states_raw: pd.DataFrame,
    states_norm: pd.DataFrame,
    actions_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    feature_cols: List[str],
    action_cols: List[str],
    mask_cols: List[str],
    delta_cols: List[str],
    reward_col: str,
    outcome_col: str,
    action_cutoffs: Tuple[List[float], List[float]],
    num_actions: int = 25,
    min_seq_len: int = 2,
) -> Dict[str, np.ndarray] | None:
    stay_states_raw = states_raw[states_raw[C_ICUSTAYID] == stay_id].copy()
    stay_states_norm = states_norm[states_norm[C_ICUSTAYID] == stay_id].copy()
    stay_actions = actions_df[actions_df[C_ICUSTAYID] == stay_id].copy()
    stay_mask = mask_df[mask_df[C_ICUSTAYID] == stay_id].copy()
    stay_delta = delta_df[delta_df[C_ICUSTAYID] == stay_id].copy()

    if len(stay_states_raw) == 0:
        return None

    stay_states_raw = stay_states_raw.sort_values(C_TIMESTEP).reset_index(drop=True)
    stay_states_norm = stay_states_norm.sort_values(C_TIMESTEP).reset_index(drop=True)
    stay_actions = stay_actions.sort_values(C_TIMESTEP).reset_index(drop=True)
    stay_mask = stay_mask.sort_values(C_TIMESTEP).reset_index(drop=True)
    stay_delta = stay_delta.sort_values(C_TIMESTEP).reset_index(drop=True)

    if len(stay_states_raw) < min_seq_len:
        return None

    state_ts = stay_states_raw[C_TIMESTEP].values
    norm_ts = stay_states_norm[C_TIMESTEP].values
    mask_ts = stay_mask[C_TIMESTEP].values
    delta_ts = stay_delta[C_TIMESTEP].values
    action_ts = stay_actions[C_TIMESTEP].values

    if not np.array_equal(state_ts, norm_ts):
        raise ValueError(f"Timestep mismatch between raw/norm states for stay {stay_id}")
    if not np.array_equal(state_ts, mask_ts):
        raise ValueError(f"Timestep mismatch between states/mask for stay {stay_id}")
    if not np.array_equal(state_ts, delta_ts):
        raise ValueError(f"Timestep mismatch between states/delta for stay {stay_id}")

    # We keep all states.
    # Actions are expected either:
    #   1) on the first T-1 state timesteps (preferred; we then shift with no-op at t=0)
    #   2) already length T and aligned to states (fallback)
    if len(action_ts) == len(state_ts) - 1 and np.array_equal(action_ts, state_ts[:-1]):
        action_input = stay_actions[action_cols[0]].astype(np.float32).fillna(0.0).values
        action_vaso = stay_actions[action_cols[1]].astype(np.float32).fillna(0.0).values
        action_ids_raw = transform_actions(action_input, action_vaso, action_cutoffs)

        no_op_id = 0
        action_ids = np.concatenate([[no_op_id], action_ids_raw], axis=0)
    elif len(action_ts) == len(state_ts) and np.array_equal(action_ts, state_ts):
        action_input = stay_actions[action_cols[0]].astype(np.float32).fillna(0.0).values
        action_vaso = stay_actions[action_cols[1]].astype(np.float32).fillna(0.0).values
        action_ids = transform_actions(action_input, action_vaso, action_cutoffs)
    else:
        raise ValueError(
            f"Unexpected state/action timestep pattern for stay {stay_id}. "
            f"states={len(state_ts)}, actions={len(action_ts)}"
        )

    if len(action_ids) != len(state_ts):
        raise ValueError(
            f"After alignment, action length != state length for stay {stay_id}: "
            f"{len(action_ids)} vs {len(state_ts)}"
        )

    features = stay_states_norm[feature_cols].astype(np.float32).values
    mask = stay_mask[mask_cols].astype(np.float32).fillna(0.0).values
    delta = stay_delta[delta_cols].astype(np.float32).fillna(0.0).values
    reward = stay_states_raw[reward_col].astype(np.float32).fillna(0.0).values
    timesteps = stay_states_raw[C_TIMESTEP].astype(np.float32).values

    if outcome_col not in stay_states_raw.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in states file")
    outcome_value = float(stay_states_raw[outcome_col].iloc[0])
    mortality = np.full(len(state_ts), outcome_value, dtype=np.float32)

    is_first = build_is_first(len(state_ts)).astype(np.float32)
    is_terminal = build_is_terminal(len(state_ts)).astype(np.float32)
    discount = build_discount_sequence(len(state_ts)).astype(np.float32)

    action = one_hot_actions(action_ids, num_actions=num_actions)

    episode = {
        "icustayid": np.array([stay_id], dtype=np.int64),
        "timestep": timesteps,
        "features": features.astype(np.float32),
        "action": action.astype(np.float32),
        "reward": reward.astype(np.float32),
        "mask": mask.astype(np.float32),
        "delta": delta.astype(np.float32),
        "is_first": is_first,
        "is_terminal": is_terminal,
        "discount": discount,
        "mortality": mortality,
    }

    return episode


def save_episodes(
    episodes_dir: str,
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
    outcome_col: str,
    action_cutoffs: Tuple[List[float], List[float]],
    num_actions: int,
):
    saved = 0
    skipped = 0

    for stay_id in stay_ids:
        episode = prepare_episode_for_stay(
            stay_id=stay_id,
            states_raw=states_raw,
            states_norm=states_norm,
            actions_df=actions_df,
            mask_df=mask_df,
            delta_df=delta_df,
            feature_cols=feature_cols,
            action_cols=action_cols,
            mask_cols=mask_cols,
            delta_cols=delta_cols,
            reward_col=reward_col,
            outcome_col=outcome_col,
            action_cutoffs=action_cutoffs,
            num_actions=num_actions,
            min_seq_len=2,
        )

        if episode is None:
            skipped += 1
            continue

        out_path = os.path.join(episodes_dir, f"{int(stay_id)}.npz")
        save_npz(out_path, episode)
        saved += 1

    print(f"Saved {saved} episodes to {episodes_dir}. Skipped {skipped} stays.")


def main():
    parser = argparse.ArgumentParser(
        description="Build MedDreamer episodes (.npz per stay) from separate CSV files"
    )
    parser.add_argument("--states", type=str, required=True, help="Path to patient_states_with_reward.csv")
    parser.add_argument("--actions", type=str, required=True, help="Path to actions_filtered.csv")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask_filtered.csv")
    parser.add_argument("--delta", type=str, required=True, help="Path to delta_filtered.csv")
    parser.add_argument("--cohort", type=str, default=None, help="Optional path to sepsis_cohort.csv")
    parser.add_argument("--output", type=str, required=True, help="Root output dir, e.g. data/final_dataset")
    parser.add_argument("--reward-col", type=str, default=C_REWARD, help="Reward column name in states file")
    parser.add_argument("--outcome-col", type=str, default=C_MORTA_90, help="Outcome column name in states file")
    parser.add_argument("--action-cols", type=str, default=None, help="Comma-separated action columns from actions file")
    parser.add_argument("--num-action-bins", type=int, default=5, help="Per-dimension action bins; 5 -> 25 total actions")
    args = parser.parse_args()

    dataset_name = "mimic"
    num_actions = args.num_action_bins * args.num_action_bins

    dataset_dir = os.path.join(args.output, dataset_name)
    episodes_dir = os.path.join(dataset_dir, "episodes")
    ensure_dir(dataset_dir)
    ensure_dir(episodes_dir)

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

    check_mask_delta_alignment(states_df, mask_df, delta_df)

    required_state_cols = [C_ICUSTAYID, C_TIMESTEP, args.reward_col, args.outcome_col]
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

    if len(feature_cols) != 40:
        print(f"Warning: found {len(feature_cols)} feature columns, config expects 40 for sepsis.")

    if args.action_cols is None:
        action_cols = infer_default_action_columns(actions_df)
    else:
        action_cols = [x.strip() for x in args.action_cols.split(",") if x.strip()]

    validate_columns_exist(actions_df, action_cols, "action")
    if len(action_cols) != 2:
        raise ValueError("For sepsis, action-cols must contain exactly 2 columns: fluids and vasopressor.")

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

    print("Fitting normalization on all cohort states ...")
    # We use all exported episodes later, and MedDreamer will split internally.
    # So action binning / normalization here are dataset-level preprocessing.
    normer = DataNormalization(states_df)

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

    print("Fitting action bins on all cohort actions ...")
    all_action_ids, action_medians, action_cutoffs = fit_action_bins(
        actions_df[action_cols[0]].astype(np.float32).fillna(0.0).values,
        actions_df[action_cols[1]].astype(np.float32).fillna(0.0).values,
        n_action_bins=args.num_action_bins,
    )
    print(f"Total discrete actions: {num_actions}")
    print(f"Unique action ids found: {sorted(np.unique(all_action_ids).tolist())[:10]} ...")

    normer_path = os.path.join(dataset_dir, "normalization.pkl")
    normer.save(normer_path)

    column_config = {
        "dataset": dataset_name,
        "feature_cols": feature_cols,
        "action_cols_continuous": action_cols,
        "mask_cols": mask_cols,
        "delta_cols": delta_cols,
        "reward_col": args.reward_col,
        "outcome_col": args.outcome_col,
        "num_action_bins": args.num_action_bins,
        "num_actions": num_actions,
        "state_columns_declared": STATE_COLUMNS,
        "all_feature_columns_declared": ALL_FEATURE_COLUMNS,
    }
    save_pickle(os.path.join(dataset_dir, "column_config.pkl"), column_config)

    action_config = {
        "num_action_bins": args.num_action_bins,
        "num_actions": num_actions,
        "action_cols_continuous": action_cols,
        "action_cutoffs": action_cutoffs,
        "action_medians": action_medians,
        "no_op_action_id": 0,
    }
    save_pickle(os.path.join(dataset_dir, "action_config.pkl"), action_config)

    save_episodes(
        episodes_dir=episodes_dir,
        states_raw=states_df,
        states_norm=states_norm_df,
        actions_df=actions_df,
        mask_df=mask_df,
        delta_df=delta_df,
        stay_ids=np.array(sorted(stay_ids)),
        feature_cols=feature_cols,
        action_cols=action_cols,
        mask_cols=mask_cols,
        delta_cols=delta_cols,
        reward_col=args.reward_col,
        outcome_col=args.outcome_col,
        action_cutoffs=action_cutoffs,
        num_actions=num_actions,
    )

    print("Done.")


if __name__ == "__main__":
    main()