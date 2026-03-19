import os
import argparse
import numpy as np


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


def check_episode(ep: dict, expected_num_actions: int | None = None) -> list[str]:
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

        if not np.all(np.diff(ep["timestep"]) >= 0):
            errors.append("timesteps are not sorted non-decreasingly")

    for key, value in ep.items():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            if np.isnan(value).any():
                errors.append(f"{key} contains NaN")
            if np.isinf(value).any():
                errors.append(f"{key} contains Inf")

    return errors


def summarize_episode(ep: dict) -> dict:
    action_ids = np.argmax(ep["action"], axis=1)
    return {
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
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MedDreamer episode .npz files"
    )
    parser.add_argument(
        "--episodes-dir",
        type=str,
        required=True,
        help="Path to episodes directory, e.g. data/meddreamer_dataset/mimic/episodes",
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
    all_action_ids = set()

    for i, path in enumerate(files):
        ep = load_episode(path)
        errors = check_episode(ep, expected_num_actions=args.num_actions)

        if errors:
            bad_files.append((path, errors))
            continue

        summary = summarize_episode(ep)
        lengths.append(summary["T"])
        feature_dims.append(summary["num_features"])
        reward_sums.append(summary["reward_sum"])
        mortalities.append(summary["mortality"])
        all_action_ids.update(summary["unique_actions"])

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

    print("\n" + "=" * 80)
    print("GLOBAL SUMMARY")
    print("=" * 80)

    valid_count = len(files) - len(bad_files)
    print(f"Valid episodes: {valid_count}")
    print(f"Invalid episodes: {len(bad_files)}")

    if valid_count > 0:
        print(f"Min length: {min(lengths)}")
        print(f"Mean length: {np.mean(lengths):.2f}")
        print(f"Max length: {max(lengths)}")
        print(f"Feature dims found: {sorted(set(feature_dims))}")
        print(f"Reward sum mean: {np.mean(reward_sums):.4f}")
        print(f"Reward sum min: {np.min(reward_sums):.4f}")
        print(f"Reward sum max: {np.max(reward_sums):.4f}")
        print(f"Mortality mean: {np.mean(mortalities):.4f}")
        print(f"Observed action ids: {sorted(all_action_ids)}")

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