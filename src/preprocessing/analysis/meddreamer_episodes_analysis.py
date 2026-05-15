import os
import argparse
import numpy as np

from src.preprocessing.utils import load_csv
from src.preprocessing.columns import C_SUBJECT_ID, C_ICUSTAYID, C_RE_ADMISSION


REQUIRED_KEYS = [
    "icustayid",
    "timestep",
    "features",
    "action",
    "reward",
    "mask",
    "delta",
    "is_first",
    "is_terminal",
    "discount",
    "mortality",
]

OPTIONAL_KEYS = [
    "sofa",
    "cont",
]


def load_episode(path: str) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def check_one_hot(actions: np.ndarray) -> tuple[bool, str]:
    if actions.ndim != 2:
        return False, f"action ndim is {actions.ndim}, expected 2"

    row_sums = actions.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        return False, "some action rows do not sum to 1"

    if not np.all((actions == 0.0) | (actions == 1.0)):
        return False, "action matrix is not binary one-hot"

    return True, "ok"


def derive_expected_cont_from_episode(ep: dict):
    """
    Reconstruct the 'cont' target that WorldModel.preprocess() would build
    starting from episode fields.

    Returns:
        expected_type: one of {"cont", "mort2", "mort3"}
        expected_cont: np.ndarray with shape [T], [T,1], or [T,3]
    """
    is_terminal = ep["is_terminal"].astype(np.float32)
    mortality = ep["mortality"].astype(np.float32)
    T = len(is_terminal)

    # cont_type == "cont"
    expected_cont = 1.0 - is_terminal
    return "cont", expected_cont


def check_cont(ep: dict) -> list[str]:
    """
    Validate cont only if it already exists in the saved episode.
    Since in your pipeline cont is normally built on-the-fly in preprocess(),
    this check is optional and consistency-based.
    """
    errors = []

    if "cont" not in ep:
        return errors

    cont = ep["cont"]
    T = len(ep["timestep"])

    if cont.shape[0] != T:
        errors.append(f"cont first dimension {cont.shape[0]} != timestep length {T}")
        return errors

    expected_type, expected_cont = derive_expected_cont_from_episode(ep)

    if cont.ndim == 1:
        if not np.all((cont == 0.0) | (cont == 1.0)):
            errors.append("cont (1D) is not binary (contains values different from 0/1)")

        if expected_type == "cont":
            if not np.array_equal(cont.astype(np.float32), expected_cont.astype(np.float32)):
                errors.append("cont values are not consistent with (1 - is_terminal)")

    elif cont.ndim == 2 and cont.shape[1] == 1:
        cont_flat = cont[:, 0]
        if not np.all((cont_flat == 0.0) | (cont_flat == 1.0)):
            errors.append("cont ([T,1]) is not binary (contains values different from 0/1)")

        if expected_type == "cont":
            if not np.array_equal(cont_flat.astype(np.float32), expected_cont.astype(np.float32)):
                errors.append("cont[:,0] values are not consistent with (1 - is_terminal)")

    elif cont.ndim == 2 and cont.shape[1] == 3:
        row_sums = cont.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            errors.append("cont ([T,3]) rows do not sum to 1")

        if not np.all((cont == 0.0) | (cont == 1.0)):
            errors.append("cont ([T,3]) is not one-hot binary")

        last_class = int(np.argmax(cont[-1]))
        if ep["is_terminal"][-1] == 1 and last_class == 2:
            errors.append(
                "cont last row predicts class 2 (continue/ICU) but last is_terminal is 1"
            )
        if ep["is_terminal"][-1] == 0 and last_class != 2:
            errors.append(
                "cont last row predicts terminal class but last is_terminal is 0"
            )

    else:
        errors.append(
            f"cont has unsupported shape {cont.shape}; expected [T], [T,1], or [T,3]"
        )

    return errors


def check_episode(
    ep: dict,
    expected_num_actions: int | None = None,
    allow_sofa_nan: bool = True,
) -> list[str]:
    errors = []

    for key in REQUIRED_KEYS:
        if key not in ep:
            errors.append(f"missing key: {key}")

    if errors:
        return errors

    T = len(ep["timestep"])

    for key in ["features", "action", "mask", "delta"]:
        if ep[key].shape[0] != T:
            errors.append(f"{key} first dimension {ep[key].shape[0]} != timestep length {T}")

    for key in ["reward", "is_first", "is_terminal", "discount", "mortality"]:
        if len(ep[key]) != T:
            errors.append(f"{key} length {len(ep[key])} != timestep length {T}")

    if ep["features"].ndim != 2:
        errors.append(f"features ndim is {ep['features'].ndim}, expected 2")

    if ep["mask"].ndim != 2:
        errors.append(f"mask ndim is {ep['mask'].ndim}, expected 2")

    if not np.all((ep["mask"] == 0.0) | (ep["mask"] == 1.0)):
        errors.append("mask is not binary (contains values different from 0/1)")

    if ep["delta"].ndim != 2:
        errors.append(f"delta ndim is {ep['delta'].ndim}, expected 2")

    if ep["features"].shape != ep["mask"].shape:
        errors.append(f"features shape {ep['features'].shape} != mask shape {ep['mask'].shape}")

    if ep["features"].shape != ep["delta"].shape:
        errors.append(f"features shape {ep['features'].shape} != delta shape {ep['delta'].shape}")

    ok_onehot, msg_onehot = check_one_hot(ep["action"])
    if not ok_onehot:
        errors.append(f"action error: {msg_onehot}")

    if expected_num_actions is not None and ep["action"].shape[1] != expected_num_actions:
        errors.append(
            f"action second dimension {ep['action'].shape[1]} != expected {expected_num_actions}"
        )

    if T > 0:
        if ep["is_first"][0] != 1:
            errors.append("is_first[0] is not 1")

        if np.sum(ep["is_first"]) != 1:
            errors.append("is_first should contain exactly one 1")

        if ep["is_terminal"][-1] != 1:
            errors.append("last is_terminal is not 1")

        if np.sum(ep["is_terminal"]) != 1:
            errors.append("is_terminal should contain exactly one 1")

        if ep["discount"][-1] != 0:
            errors.append("last discount is not 0")

        if not np.all(ep["discount"][:-1] == 1):
            errors.append("all discount values except the last should be 1")

        if not np.all(np.diff(ep["timestep"]) >= 0):
            errors.append("timesteps are not sorted non-decreasingly")

    for key, value in ep.items():
        if not isinstance(value, np.ndarray):
            continue
        if not np.issubdtype(value.dtype, np.number):
            continue

        if key == "sofa" and allow_sofa_nan:
            if np.isinf(value).any():
                errors.append("sofa contains Inf")
            continue

        if np.isnan(value).any():
            errors.append(f"{key} contains NaN")

        if np.isinf(value).any():
            errors.append(f"{key} contains Inf")

    if "sofa" in ep:
        if len(ep["sofa"]) != T:
            errors.append(f"sofa length {len(ep['sofa'])} != timestep length {T}")

        if ep["sofa"].ndim != 1:
            errors.append(f"sofa ndim is {ep['sofa'].ndim}, expected 1")

    errors.extend(check_cont(ep))

    return errors


def summarize_episode(ep: dict) -> dict:
    action_ids = np.argmax(ep["action"], axis=1)

    summary = {
        "stay_id": int(ep["icustayid"][0]) if len(ep["icustayid"]) > 0 else None,
        "T": len(ep["timestep"]),
        "num_features": ep["features"].shape[1],
        "num_actions": ep["action"].shape[1],
        "reward_sum": float(ep["reward"].sum()),
        "reward_min": float(ep["reward"].min()),
        "reward_max": float(ep["reward"].max()),
        "mortality": float(ep["mortality"][0]) if len(ep["mortality"]) > 0 else None,
        "action_min": int(action_ids.min()),
        "action_max": int(action_ids.max()),
        "unique_actions": sorted(np.unique(action_ids).tolist()),
        "mask_observed_fraction": float(ep["mask"].mean()),
        "delta_mean": float(ep["delta"].mean()),
        "is_terminal_sum": int(np.sum(ep["is_terminal"])),
        "last_is_terminal": float(ep["is_terminal"][-1]),
        "last_discount": float(ep["discount"][-1]),
    }

    if "sofa" in ep:
        sofa = ep["sofa"]
        sofa_valid = sofa[~np.isnan(sofa)]

        summary["sofa_missing_fraction"] = float(np.isnan(sofa).mean())

        if len(sofa_valid) > 0:
            summary["sofa_min"] = float(np.min(sofa_valid))
            summary["sofa_max"] = float(np.max(sofa_valid))
            summary["sofa_first_valid"] = float(sofa_valid[0])
            summary["sofa_last_valid"] = float(sofa_valid[-1])
        else:
            summary["sofa_min"] = None
            summary["sofa_max"] = None
            summary["sofa_first_valid"] = None
            summary["sofa_last_valid"] = None

    if "cont" in ep:
        cont = ep["cont"]
        summary["cont_shape"] = tuple(cont.shape)

        if cont.ndim == 1:
            summary["cont_last"] = float(cont[-1])
            summary["cont_zero_frac"] = float((cont == 0).mean())
            summary["cont_one_frac"] = float((cont == 1).mean())

        elif cont.ndim == 2 and cont.shape[1] == 1:
            cont_flat = cont[:, 0]
            summary["cont_last"] = float(cont_flat[-1])
            summary["cont_zero_frac"] = float((cont_flat == 0).mean())
            summary["cont_one_frac"] = float((cont_flat == 1).mean())

        elif cont.ndim == 2 and cont.shape[1] == 3:
            cont_cls = np.argmax(cont, axis=1)
            summary["cont_last_class"] = int(cont_cls[-1])
            summary["cont_class_counts"] = {
                int(c): int((cont_cls == c).sum()) for c in np.unique(cont_cls)
            }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MedDreamer episode .npz files"
    )
    parser.add_argument(
        "--episodes-dir",
        type=str,
        default="data/meddreamer_dataset/mimic/episodes",
        help="Path to episodes directory",
    )
    parser.add_argument(
        "--num-actions",
        type=int,
        default=25,
        help="Expected action dimension",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=3,
        help="How many example episodes to print",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of files to analyze",
    )
    parser.add_argument(
        "--demog-file",
        type=str,
        default=None,
        help="Optional path to demog.csv for reporting patient/stay counts restricted to valid episodes",
    )
    parser.add_argument(
        "--states-file",
        type=str,
        default=None,
        help="Optional path to states CSV for reporting readmission counts",
    )
    parser.add_argument(
        "--strict-sofa",
        action="store_true",
        help="If set, NaNs in sofa are treated as errors",
    )

    args = parser.parse_args()

    files = sorted(
        [
            os.path.join(args.episodes_dir, f)
            for f in os.listdir(args.episodes_dir)
            if f.endswith(".npz")
        ]
    )

    if args.max_files is not None:
        files = files[:args.max_files]

    if len(files) == 0:
        raise ValueError(f"No .npz files found in {args.episodes_dir}")

    print(f"Found {len(files)} episode files")

    bad_files = []
    lengths = []
    feature_dims = []
    reward_sums = []
    mortalities = []
    sofa_missing_fractions = []
    all_action_ids = set()
    valid_episode_stay_ids = set()

    for i, path in enumerate(files):
        ep = load_episode(path)
        errors = check_episode(
            ep,
            expected_num_actions=args.num_actions,
            allow_sofa_nan=not args.strict_sofa,
        )

        if errors:
            bad_files.append((path, errors))
            continue

        summary = summarize_episode(ep)
        valid_episode_stay_ids.add(summary["stay_id"])

        lengths.append(summary["T"])
        feature_dims.append(summary["num_features"])
        reward_sums.append(summary["reward_sum"])
        mortalities.append(summary["mortality"])
        all_action_ids.update(summary["unique_actions"])

        if "sofa_missing_fraction" in summary:
            sofa_missing_fractions.append(summary["sofa_missing_fraction"])

        if i < args.show:
            print("\n" + "=" * 80)
            print(f"EXAMPLE EPISODE {i + 1}")
            print("=" * 80)
            for k, v in summary.items():
                print(f"{k}: {v}")

            action_ids = np.argmax(ep["action"], axis=1)
            print("first 10 timesteps:", ep["timestep"][:10].tolist())
            print("first 10 action ids:", action_ids[:10].tolist())
            print("first 10 rewards:", ep["reward"][:10].tolist())
            print("first 10 is_first:", ep["is_first"][:10].tolist())
            print("first 10 is_terminal:", ep["is_terminal"][:10].tolist())
            print("first 10 discount:", ep["discount"][:10].tolist())

            if "cont" in ep:
                print("first 10 cont:", ep["cont"][:10].tolist())

            if "sofa" in ep:
                print("first 10 sofa:", ep["sofa"][:10].tolist())

    if args.demog_file is not None:
        print("\n" + "=" * 80)
        print("DATASET SUMMARY (RESTRICTED TO VALID EPISODES)")
        print("=" * 80)

        demog = load_csv(args.demog_file)

        needed_demog_cols = [C_SUBJECT_ID, C_ICUSTAYID]
        for col in needed_demog_cols:
            if col not in demog.columns:
                raise ValueError(f"Missing column in demog file: {col}")

        demog_small = demog[[C_SUBJECT_ID, C_ICUSTAYID]].drop_duplicates()
        demog_small = demog_small[demog_small[C_ICUSTAYID].isin(valid_episode_stay_ids)].copy()

        total_patients = demog_small[C_SUBJECT_ID].nunique()
        total_stays = demog_small[C_ICUSTAYID].nunique()

        print(f"Total unique patients in valid episodes: {total_patients}")
        print(f"Total unique ICU stays in valid episodes: {total_stays}")

        if args.states_file is not None:
            states = load_csv(args.states_file)

            needed_state_cols = [C_ICUSTAYID, C_RE_ADMISSION]
            for col in needed_state_cols:
                if col not in states.columns:
                    raise ValueError(f"Missing column in states file: {col}")

            states_small = states[[C_ICUSTAYID, C_RE_ADMISSION]].drop_duplicates()
            states_small = states_small[states_small[C_ICUSTAYID].isin(valid_episode_stay_ids)].copy()

            merged = demog_small.merge(states_small, on=C_ICUSTAYID, how="left")

            total_readmission_stays = merged.loc[
                merged[C_RE_ADMISSION] == 1, C_ICUSTAYID
            ].nunique()

            total_readmission_patients = merged.loc[
                merged[C_RE_ADMISSION] == 1, C_SUBJECT_ID
            ].nunique()

            print(f"Total ICU stays with readmission = 1: {total_readmission_stays}")
            print(f"Total patients with at least one readmission = 1 stay: {total_readmission_patients}")

    print("\n" + "=" * 80)
    print("GLOBAL SUMMARY")
    print("=" * 80)

    valid_count = len(files) - len(bad_files)
    print(f"Valid episodes: {valid_count}")
    print(f"Invalid episodes: {len(bad_files)}")

    if valid_count > 0:
        print(f"Unique valid ICU stays: {len(valid_episode_stay_ids)}")
        print(f"Min length: {min(lengths)}")
        print(f"Mean length: {np.mean(lengths):.2f}")
        print(f"Max length: {max(lengths)}")
        print(f"Feature dims found: {sorted(set(feature_dims))}")
        print(f"Reward sum mean: {np.mean(reward_sums):.4f}")
        print(f"Reward sum min: {np.min(reward_sums):.4f}")
        print(f"Reward sum max: {np.max(reward_sums):.4f}")
        print(f"Mortality mean: {np.mean(mortalities):.4f}")
        print(f"Observed action ids: {sorted(all_action_ids)}")

        if len(sofa_missing_fractions) > 0:
            print(f"Mean SOFA missing fraction: {np.mean(sofa_missing_fractions):.4f}")
            print(f"Min SOFA missing fraction: {np.min(sofa_missing_fractions):.4f}")
            print(f"Max SOFA missing fraction: {np.max(sofa_missing_fractions):.4f}")

    if bad_files:
        print("\n" + "=" * 80)
        print("INVALID FILES")
        print("=" * 80)
        for path, errors in bad_files[:20]:
            print(f"\n{path}")
            for err in errors:
                print(f" - {err}")


if __name__ == "__main__":
    main()