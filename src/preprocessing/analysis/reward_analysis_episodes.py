"""
python -u -m src.preprocessing.analysis.reward_analysis_episodes \
  --episodes-dir data/meddreamer_dataset/mimic/episodes \
  --output-dir reward_analysis \
  --bins 100
"""
import os
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.meddreamer.utils.tools import plot_mortality_vs_value


def load_episode(path: str) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def safe_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q01": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "max": np.nan,
        }

    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
    }


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def count_special_values(
    rewards: np.ndarray,
    terminal_abs_value: float = 15.0,
    atol: float = 1e-6,
) -> Dict[str, int]:
    rewards = np.asarray(rewards, dtype=np.float32)

    return {
        "num_zero": int(np.sum(np.isclose(rewards, 0.0, atol=atol))),
        "num_positive": int(np.sum(rewards > 0.0)),
        "num_negative": int(np.sum(rewards < 0.0)),
        "num_pos_terminal": int(np.sum(np.isclose(rewards, terminal_abs_value, atol=atol))),
        "num_neg_terminal": int(np.sum(np.isclose(rewards, -terminal_abs_value, atol=atol))),
    }


def summarize_group(
    rewards: np.ndarray,
    terminal_abs_value: float = 15.0,
) -> Dict:
    rewards = np.asarray(rewards, dtype=np.float32)

    out = {}
    out["stats"] = safe_stats(rewards)
    out["counts"] = count_special_values(rewards, terminal_abs_value=terminal_abs_value)

    n = len(rewards)
    if n > 0:
        out["fractions"] = {
            "frac_zero": float(out["counts"]["num_zero"] / n),
            "frac_positive": float(out["counts"]["num_positive"] / n),
            "frac_negative": float(out["counts"]["num_negative"] / n),
            "frac_pos_terminal": float(out["counts"]["num_pos_terminal"] / n),
            "frac_neg_terminal": float(out["counts"]["num_neg_terminal"] / n),
        }
    else:
        out["fractions"] = {
            "frac_zero": np.nan,
            "frac_positive": np.nan,
            "frac_negative": np.nan,
            "frac_pos_terminal": np.nan,
            "frac_neg_terminal": np.nan,
        }

    return out


def summarize_by_outcome(values: np.ndarray, mortalities: np.ndarray) -> Dict:
    values = np.asarray(values, dtype=np.float32)
    mortalities = np.asarray(mortalities, dtype=np.float32)

    surv_mask = mortalities == 0
    death_mask = mortalities == 1

    surv_values = values[surv_mask]
    death_values = values[death_mask]

    surv_mean = float(np.mean(surv_values)) if len(surv_values) > 0 else np.nan
    death_mean = float(np.mean(death_values)) if len(death_values) > 0 else np.nan

    return {
        "survivors": safe_stats(surv_values),
        "deaths": safe_stats(death_values),
        "gap_survivors_minus_deaths": float(surv_mean - death_mean)
        if np.isfinite(surv_mean) and np.isfinite(death_mean)
        else np.nan,
        "corr_with_mortality": safe_corr(values, mortalities),
    }


def make_histogram(
    values: np.ndarray,
    title: str,
    xlabel: str,
    output_path: str,
    bins: int = 100,
):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        print(f"Skipping empty histogram: {output_path}")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze reward distribution and reward-outcome relation in MedDreamer episodes."
    )
    parser.add_argument(
        "--episodes-dir",
        type=str,
        required=True,
        help="Path to episodes directory, e.g. data/meddreamer_dataset/mimic/episodes",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where analysis outputs will be saved",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of episode files to analyze",
    )
    parser.add_argument(
        "--terminal-abs-value",
        type=float,
        default=15.0,
        help="Absolute value used for old terminal rewards, typically 15.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of bins for histograms",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(
        os.path.join(args.episodes_dir, f)
        for f in os.listdir(args.episodes_dir)
        if f.endswith(".npz")
    )

    if args.max_files is not None:
        files = files[:args.max_files]

    if len(files) == 0:
        raise ValueError(f"No .npz files found in {args.episodes_dir}")

    print(f"Found {len(files)} episode files")

    all_rewards: List[np.ndarray] = []
    first_rewards: List[float] = []
    intermediate_rewards: List[np.ndarray] = []
    terminal_rewards: List[float] = []

    rewards_alive: List[np.ndarray] = []
    rewards_dead: List[np.ndarray] = []

    terminal_alive: List[float] = []
    terminal_dead: List[float] = []

    episode_lengths = []
    episode_returns = []
    episode_returns_no_first = []
    episode_intermediate_returns = []
    episode_terminal_rewards = []
    episode_mortalities = []

    rows = []

    for i, path in enumerate(files):
        ep = load_episode(path)

        if "reward" not in ep:
            print(f"Skipping {path}: missing reward")
            continue
        if "mortality" not in ep:
            print(f"Skipping {path}: missing mortality")
            continue

        reward = ep["reward"].reshape(-1).astype(np.float32)
        mortality = ep["mortality"].reshape(-1).astype(np.float32)

        if len(reward) == 0:
            continue

        died = int(mortality[0]) if len(mortality) > 0 else 0

        if "icustayid" in ep and len(ep["icustayid"]) > 0:
            stay_id = int(np.asarray(ep["icustayid"]).reshape(-1)[0])
        else:
            stay_id = os.path.basename(path).replace(".npz", "")

        all_rewards.append(reward)
        first_rewards.append(float(reward[0]))
        terminal_rewards.append(float(reward[-1]))

        if len(reward) > 2:
            intermediate = reward[1:-1]
        else:
            intermediate = np.array([], dtype=np.float32)

        intermediate_rewards.append(intermediate)

        if died == 1:
            rewards_dead.append(reward)
            terminal_dead.append(float(reward[-1]))
        else:
            rewards_alive.append(reward)
            terminal_alive.append(float(reward[-1]))

        ep_return = float(np.sum(reward))
        ep_return_no_first = float(np.sum(reward[1:])) if len(reward) > 1 else 0.0
        ep_intermediate_return = float(np.sum(intermediate)) if len(intermediate) > 0 else 0.0
        ep_terminal_reward = float(reward[-1])

        episode_lengths.append(len(reward))
        episode_returns.append(ep_return)
        episode_returns_no_first.append(ep_return_no_first)
        episode_intermediate_returns.append(ep_intermediate_return)
        episode_terminal_rewards.append(ep_terminal_reward)
        episode_mortalities.append(died)

        rows.append(
            {
                "stay_id": stay_id,
                "T": len(reward),
                "mortality": died,
                "episode_return": ep_return,
                "episode_return_no_first": ep_return_no_first,
                "episode_intermediate_return": ep_intermediate_return,
                "terminal_reward": ep_terminal_reward,
                "first_reward": float(reward[0]),
            }
        )

        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1}/{len(files)} episodes")

    all_rewards_np = (
        np.concatenate(all_rewards) if len(all_rewards) > 0 else np.array([], dtype=np.float32)
    )
    first_rewards_np = np.array(first_rewards, dtype=np.float32)
    terminal_rewards_np = np.array(terminal_rewards, dtype=np.float32)
    intermediate_rewards_np = (
        np.concatenate([x for x in intermediate_rewards if len(x) > 0])
        if any(len(x) > 0 for x in intermediate_rewards)
        else np.array([], dtype=np.float32)
    )

    rewards_alive_np = (
        np.concatenate(rewards_alive) if len(rewards_alive) > 0 else np.array([], dtype=np.float32)
    )
    rewards_dead_np = (
        np.concatenate(rewards_dead) if len(rewards_dead) > 0 else np.array([], dtype=np.float32)
    )

    terminal_alive_np = np.array(terminal_alive, dtype=np.float32)
    terminal_dead_np = np.array(terminal_dead, dtype=np.float32)

    episode_lengths_np = np.array(episode_lengths, dtype=np.float32)
    episode_returns_np = np.array(episode_returns, dtype=np.float32)
    episode_returns_no_first_np = np.array(episode_returns_no_first, dtype=np.float32)
    episode_intermediate_returns_np = np.array(episode_intermediate_returns, dtype=np.float32)
    episode_terminal_rewards_np = np.array(episode_terminal_rewards, dtype=np.float32)
    episode_mortalities_np = np.array(episode_mortalities, dtype=np.float32)

    summary = {
        "num_episodes": int(len(episode_lengths)),
        "mortality_rate": float(np.mean(episode_mortalities_np)) if len(episode_mortalities_np) > 0 else np.nan,
        "episode_length_stats": safe_stats(episode_lengths_np),
        "episode_return_stats": safe_stats(episode_returns_np),
        "episode_return_no_first_stats": safe_stats(episode_returns_no_first_np),
        "episode_intermediate_return_stats": safe_stats(episode_intermediate_returns_np),
        "episode_terminal_reward_stats": safe_stats(episode_terminal_rewards_np),
        "all_rewards": summarize_group(all_rewards_np, terminal_abs_value=args.terminal_abs_value),
        "first_rewards": summarize_group(first_rewards_np, terminal_abs_value=args.terminal_abs_value),
        "intermediate_rewards": summarize_group(intermediate_rewards_np, terminal_abs_value=args.terminal_abs_value),
        "terminal_rewards": summarize_group(terminal_rewards_np, terminal_abs_value=args.terminal_abs_value),
        "rewards_alive": summarize_group(rewards_alive_np, terminal_abs_value=args.terminal_abs_value),
        "rewards_dead": summarize_group(rewards_dead_np, terminal_abs_value=args.terminal_abs_value),
        "terminal_alive": summarize_group(terminal_alive_np, terminal_abs_value=args.terminal_abs_value),
        "terminal_dead": summarize_group(terminal_dead_np, terminal_abs_value=args.terminal_abs_value),
        "episode_return_by_outcome": summarize_by_outcome(episode_returns_np, episode_mortalities_np),
        "episode_return_no_first_by_outcome": summarize_by_outcome(
            episode_returns_no_first_np,
            episode_mortalities_np,
        ),
        "episode_intermediate_return_by_outcome": summarize_by_outcome(
            episode_intermediate_returns_np,
            episode_mortalities_np,
        ),
        "episode_terminal_reward_by_outcome": summarize_by_outcome(
            episode_terminal_rewards_np,
            episode_mortalities_np,
        ),
        "episode_length_by_outcome": summarize_by_outcome(
            episode_lengths_np,
            episode_mortalities_np,
        ),
    }

    episode_csv = os.path.join(args.output_dir, "episode_reward_outcome.csv")
    pd.DataFrame(rows).to_csv(episode_csv, index=False)

    summary_json = os.path.join(args.output_dir, "reward_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    make_histogram(
        all_rewards_np,
        "All rewards",
        "Reward",
        os.path.join(args.output_dir, "hist_all_rewards.png"),
        bins=args.bins,
    )
    make_histogram(
        intermediate_rewards_np,
        "Intermediate rewards",
        "Reward",
        os.path.join(args.output_dir, "hist_intermediate_rewards.png"),
        bins=args.bins,
    )
    make_histogram(
        terminal_rewards_np,
        "Terminal rewards",
        "Reward",
        os.path.join(args.output_dir, "hist_terminal_rewards.png"),
        bins=args.bins,
    )
    make_histogram(
        episode_returns_np,
        "Episode returns",
        "Return",
        os.path.join(args.output_dir, "hist_episode_returns.png"),
        bins=args.bins,
    )
    make_histogram(
        episode_returns_no_first_np,
        "Episode returns excluding first step",
        "Return excluding first reward",
        os.path.join(args.output_dir, "hist_episode_returns_no_first.png"),
        bins=args.bins,
    )
    make_histogram(
        episode_intermediate_returns_np,
        "Episode intermediate returns",
        "Intermediate return",
        os.path.join(args.output_dir, "hist_episode_intermediate_returns.png"),
        bins=args.bins,
    )

    for values, xlabel, fname in [
        (episode_returns_np,              "Episode Return",           "mortality_vs_episode_return.png"),
        (episode_returns_no_first_np,     "Episode Return (no first)","mortality_vs_episode_return_no_first.png"),
        (episode_intermediate_returns_np, "Intermediate Return",      "mortality_vs_intermediate_return.png"),
        (episode_terminal_rewards_np,     "Terminal Reward",          "mortality_vs_terminal_reward.png"),
    ]:
        fig, _, _, _ = plot_mortality_vs_value(values, episode_mortalities_np, xlabel=xlabel)
        fig.savefig(os.path.join(args.output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"Saved: {fname}")

    print("\n=== REWARD ANALYSIS SUMMARY ===")
    print(f"Episodes analyzed: {len(episode_lengths)}")
    print(f"Mortality rate: {summary['mortality_rate']:.6f}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved episode CSV: {episode_csv}")

    print("\nEpisode return by outcome:")
    print(json.dumps(summary["episode_return_by_outcome"], indent=2))

    print("\nEpisode return excluding first reward by outcome:")
    print(json.dumps(summary["episode_return_no_first_by_outcome"], indent=2))

    print("\nIntermediate return by outcome:")
    print(json.dumps(summary["episode_intermediate_return_by_outcome"], indent=2))

    print("\nTerminal reward by outcome:")
    print(json.dumps(summary["episode_terminal_reward_by_outcome"], indent=2))

    print("\nAll rewards:")
    print(json.dumps(summary["all_rewards"], indent=2))

    print("\nIntermediate rewards:")
    print(json.dumps(summary["intermediate_rewards"], indent=2))

    print("\nTerminal rewards:")
    print(json.dumps(summary["terminal_rewards"], indent=2))


if __name__ == "__main__":
    main()