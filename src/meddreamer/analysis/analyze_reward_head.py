"""
Analyze reward head calibration: compare r_true vs r_hat = reward_head(posterior).

Usage:
    python -u -m src.meddreamer.analysis.analyze_reward_head \
        --configs eval \
        --ckptdir /path/to/wm/checkpoints \
        --ckptepoch 10000 \
        --output-dir analysis_outputs/reward_head \
        --max-episodes 2000
"""
import os
import argparse
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import src.meddreamer.utils.tools as tools
from src.meddreamer.dreamer import MedDreamer


def make_dataset_and_agent(config):
    eps_dir = os.path.join(config.datadir, config.dataset, "episodes")
    all_stay_ids = tools.load_all_episode_keys(eps_dir)

    cache_root = os.path.dirname(eps_dir)
    split_path = os.path.join(cache_root, f"splits_seed{config.seed}.pkl")

    with open(split_path, "rb") as f:
        splits = pickle.load(f)

    test_cache = os.path.join(cache_root, f"test_eps_cache_seed{config.seed}.pkl")
    episodes = tools.load_split_episodes(eps_dir, splits["test"], cache_path=test_cache)

    logdir = pathlib.Path("tmp_reward_head_analysis")
    logdir.mkdir(exist_ok=True)
    logger = tools.Logger(logdir)

    agent = MedDreamer(config, logger, logdir, None, episodes).to(config.device)
    tools.load_model(agent, "wm", config.ckptdir, config.ckptepoch, config.device)
    agent._wm.eval()

    return agent, episodes


def collect_mortality_vs_return(agent, episodes, max_episodes, context_steps=5):
    """Compute per-episode returns (real and WM prior) for mortality analysis.

    For each episode:
      - real_return:  sum of actual rewards from step context_steps onward
      - wm_return:    sum of reward_head(prior) predictions using real clinician
                      actions, starting from the posterior state at context_steps-1
    """
    real_returns = []
    wm_returns = []
    mortalities = []
    count = 0

    total = min(max_episodes, len(episodes)) if max_episodes is not None else len(episodes)
    with torch.no_grad():
        for stay_id, data in tqdm(episodes.items(), total=total, desc="Mortality episodes", unit="ep"):
            if max_episodes is not None and count >= max_episodes:
                break

            data = agent._expand_episode(data)
            T = data["features"].shape[1]
            if T <= context_steps:
                continue

            post, embed, data = agent._wm._load(data)

            init = {k: v[:, context_steps - 1] for k, v in post.items()}
            phys_action = data["action"][:, context_steps:]  # (B, T-ctx, A)

            prior = agent._wm.dynamics.imagine_with_action(phys_action, init)
            feat_prior = agent._wm.dynamics.get_feat(prior)              # (B, T-ctx, D)
            reward_wm = agent._wm.heads["reward"](feat_prior).mode()     # (B, T-ctx, 1)

            wm_ret = float(reward_wm.sum(dim=1).squeeze().item())
            real_ret = float(data["reward"][:, context_steps:].sum(dim=1).squeeze().item())
            mortality = float(data["mortality"][0, 0].item())

            wm_returns.append(wm_ret)
            real_returns.append(real_ret)
            mortalities.append(mortality)
            count += 1

    return np.array(real_returns, dtype=np.float32), \
           np.array(wm_returns, dtype=np.float32), \
           np.array(mortalities, dtype=np.float32)


def collect_rows(agent, episodes, max_episodes, device):
    rows = []
    count = 0

    total = min(max_episodes, len(episodes)) if max_episodes is not None else len(episodes)
    with torch.no_grad():
        for stay_id, data in tqdm(episodes.items(), total=total, desc="Episodes", unit="ep"):
            if max_episodes is not None and count >= max_episodes:
                break

            data = agent._expand_episode(data)
            if data["features"].shape[1] < 2:
                continue

            post, embed, data = agent._wm._load(data)
            feat = agent._wm.dynamics.get_feat(post)

            r_pred = agent._wm.heads["reward"](feat).mode().squeeze(-1)   # (B, T)
            r_true = data["reward"].squeeze(-1)                            # (B, T)
            is_terminal = data["is_terminal"].squeeze(-1)                  # (B, T)
            mortality = float(data["mortality"][0, 0].item())

            B, T = r_pred.shape
            for t in range(T):
                rows.append({
                    "stay_id": stay_id,
                    "t": t,
                    "r_true": float(r_true[0, t].item()),
                    "r_pred": float(r_pred[0, t].item()),
                    "is_terminal": int(is_terminal[0, t].item()),
                    "mortality": mortality,
                })

            count += 1

    return pd.DataFrame(rows)


def make_plots(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    r_true = df["r_true"].values
    r_pred = df["r_pred"].values
    is_term = df["is_terminal"].values.astype(bool)

    # histogram: r_true vs r_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, mask, label in [
        (axes[0], ~is_term, "Non-terminal steps"),
        (axes[1],  is_term, "Terminal steps"),
    ]:
        rt = r_true[mask]
        rp = r_pred[mask]
        lo = min(rt.min(), rp.min())
        hi = max(rt.max(), rp.max())
        bins = np.linspace(lo, hi, 60)
        ax.hist(rt, bins=bins, alpha=0.6, label="r_true")
        ax.hist(rp, bins=bins, alpha=0.6, label="r_pred")
        ax.set_title(label)
        ax.set_xlabel("Reward")
        ax.set_ylabel("Count")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_hist_true_vs_pred.png"), dpi=150)
    plt.close()

    # scatter r_true vs r_pred (subsample to avoid overplotting)
    idx = np.random.choice(len(r_true), size=min(5000, len(r_true)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(r_true[idx], r_pred[idx], alpha=0.2, s=5)
    lo = min(r_true[idx].min(), r_pred[idx].min())
    hi = max(r_true[idx].max(), r_pred[idx].max())
    ax.plot([lo, hi], [lo, hi], "r--", label="perfect")
    ax.set_xlabel("r_true")
    ax.set_ylabel("r_pred")
    ax.set_title("r_true vs r_pred (scatter)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_scatter.png"), dpi=150)
    plt.close()

    # MAE per group
    for label, mask in [("all", np.ones(len(df), dtype=bool)),
                         ("non_terminal", ~is_term),
                         ("terminal", is_term),
                         ("survived", df["mortality"].values == 0),
                         ("died", df["mortality"].values == 1)]:
        if mask.sum() == 0:
            continue
        mae = np.abs(r_true[mask] - r_pred[mask]).mean()
        bias = (r_pred[mask] - r_true[mask]).mean()
        print(f"[{label:15s}]  n={mask.sum():7,}  MAE={mae:.4f}  bias={bias:+.4f}")

    # error distribution: non-terminal vs terminal
    fig, ax = plt.subplots(figsize=(8, 4))
    for mask, label in [(~is_term, "non-terminal"), (is_term, "terminal")]:
        err = r_pred[mask] - r_true[mask]
        ax.hist(err, bins=60, alpha=0.6, label=label, density=True)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("r_pred - r_true  (error)")
    ax.set_ylabel("density")
    ax.set_title("Reward head error distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_error_dist.png"), dpi=150)
    plt.close()

    print(f"\nPlots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["defaults"])
    parser.add_argument("--output-dir", default="analysis_outputs/reward_head")
    parser.add_argument("--max-episodes", type=int, default=2000)
    args, remaining = parser.parse_known_args()

    yaml = YAML(typ="rt")
    configs = yaml.load(
        (pathlib.Path("src/meddreamer/configs.yaml")).read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    rem_parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        rem_parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = rem_parser.parse_args(remaining)

    tools.set_seed_everywhere(config.seed)
    agent, episodes = make_dataset_and_agent(config)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Collecting reward head predictions on {args.max_episodes} episodes...")
    df = collect_rows(agent, episodes, args.max_episodes, config.device)
    print(f"Collected {len(df):,} timesteps from {df['stay_id'].nunique()} episodes")

    csv_path = os.path.join(args.output_dir, "reward_head_calibration.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    print("\n=== REWARD HEAD CALIBRATION ===")
    make_plots(df, args.output_dir)

    print("\n=== MORTALITY VS RETURN ANALYSIS ===")
    real_returns, wm_returns, mortalities = collect_mortality_vs_return(
        agent, episodes, args.max_episodes
    )
    print(f"Episodes analyzed: {len(mortalities)}  mortality rate: {mortalities.mean():.4f}")
    print(f"Real return  — mean: {real_returns.mean():.3f}  std: {real_returns.std():.3f}")
    print(f"WM return    — mean: {wm_returns.mean():.3f}  std: {wm_returns.std():.3f}")

    fig_real, _, _, _ = tools.plot_mortality_vs_value(
        real_returns, mortalities, xlabel="Real Episode Return (sum from t=5)"
    )
    fig_real.savefig(os.path.join(args.output_dir, "mortality_vs_real_return.png"), dpi=150)
    plt.close(fig_real)

    fig_wm, _, _, _ = tools.plot_mortality_vs_value(
        wm_returns, mortalities, xlabel="WM Prior Return (reward head, sum from t=5)"
    )
    fig_wm.savefig(os.path.join(args.output_dir, "mortality_vs_wm_return.png"), dpi=150)
    plt.close(fig_wm)

    fig_both, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, vals, title in [
        (axes[0], real_returns, "Real return"),
        (axes[1], wm_returns,   "WM prior return"),
    ]:
        surv = vals[mortalities == 0]
        dead = vals[mortalities == 1]
        ax.hist(surv, bins=50, alpha=0.6, label=f"survived (n={len(surv)})", density=True)
        ax.hist(dead, bins=50, alpha=0.6, label=f"died (n={len(dead)})", density=True)
        ax.set_title(title)
        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        ax.legend()
    plt.suptitle("Episode return distribution by outcome")
    plt.tight_layout()
    fig_both.savefig(os.path.join(args.output_dir, "return_dist_by_outcome.png"), dpi=150)
    plt.close(fig_both)

    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
