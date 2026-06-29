"""
Analyze behavior policy eval CSV produced by dreamer.eval_behavior_policy().

Usage:
    python -m src.meddreamer.analysis.analyze_bp_outputs \
        --csv logs/.../bp_eval_10000.csv \
        --output-dir analysis_outputs/bp
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N_ACTIONS = 25


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def print_summary(df):
    N = len(df)
    n_actions = N_ACTIONS

    print("\n" + "=" * 70)
    print("BEHAVIOR POLICY EVALUATION")
    print("=" * 70)
    print(f"Total timesteps : {N:,}")

    print(f"\n--- Accuracy ---")
    print(f"Top-1 : {100*df['top1_hit'].mean():.2f}%")
    print(f"Top-3 : {100*df['top3_hit'].mean():.2f}%")

    print(f"\n--- Log-likelihood of clinician action ---")
    log_pi = np.log(df["pi_b_clin"].clip(lower=1e-12))
    print(f"Mean log pi_b(a_clin) : {log_pi.mean():.4f}   (random = {-np.log(n_actions):.4f})")
    print(f"Mean pi_b(a_clin)     : {df['pi_b_clin'].mean():.4f}   (uniform = {1/n_actions:.4f})")
    print(f"Median pi_b(a_clin)   : {df['pi_b_clin'].median():.4f}")

    print(f"\n--- Calibration ---")
    for thr in [0.001, 0.01, 0.05, 0.10]:
        frac = (df["pi_b_clin"] < thr).mean()
        print(f"  pi_b < {thr:.3f} : {100*frac:.1f}%  ({int(frac*N):,} steps) -> IS ratio blows up here")

    print(f"\n--- Entropy ---")
    print(f"Mean entropy   : {df['entropy'].mean():.4f}   (max = {np.log(n_actions):.4f})")
    print(f"Median entropy : {df['entropy'].median():.4f}")

    print(f"\n--- Per-action: mean pi_b(a_clin | s) ---")
    for a in range(n_actions):
        sub = df[df["clin_action"] == a]
        if len(sub) == 0:
            continue
        frac_low = (sub["pi_b_clin"] < 0.01).mean()
        print(f"  action {a:2d}  n={len(sub):6,}  mean_pi={sub['pi_b_clin'].mean():.4f}  frac<1%={100*frac_low:.1f}%")


def make_plots(df, output_dir):
    n_actions = N_ACTIONS
    pi_b = df["pi_b_clin"].values
    entropy = df["entropy"].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(pi_b, bins=50, edgecolor="k")
    axes[0].axvline(1/n_actions, color="red", linestyle="--", label=f"uniform={1/n_actions:.3f}")
    axes[0].set_xlabel("pi_b(a_clinician | s)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Distribution of pi_b for clinician action")
    axes[0].legend()

    sorted_p = np.sort(pi_b)
    axes[1].plot(sorted_p, np.linspace(0, 1, len(sorted_p)))
    axes[1].axvline(0.01, color="red", linestyle="--", label="0.01")
    axes[1].axvline(0.05, color="orange", linestyle="--", label="0.05")
    axes[1].set_xlabel("pi_b(a_clinician | s)")
    axes[1].set_ylabel("cumulative fraction")
    axes[1].set_title("CDF of pi_b(a_clinician)")
    axes[1].legend()
    axes[1].set_xlim(0, 0.5)

    axes[2].hist(entropy, bins=50, edgecolor="k")
    axes[2].axvline(np.log(n_actions), color="red", linestyle="--", label=f"max={np.log(n_actions):.2f}")
    axes[2].set_xlabel("H[pi_b(·|s)]")
    axes[2].set_ylabel("count")
    axes[2].set_title("Policy entropy per timestep")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bp_distributions.png"), dpi=200)
    plt.close()

    # per-action mean pi_b
    action_ids, mean_probs, frac_lows = [], [], []
    for a in range(n_actions):
        sub = df[df["clin_action"] == a]
        if len(sub) == 0:
            continue
        action_ids.append(a)
        mean_probs.append(sub["pi_b_clin"].mean())
        frac_lows.append((sub["pi_b_clin"] < 0.01).mean())

    x = np.arange(len(action_ids))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, mean_probs, label="mean pi_b(a|s)")
    ax.axhline(1/n_actions, color="red", linestyle="--", label=f"uniform={1/n_actions:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(action_ids)
    ax.set_xlabel("Clinician action")
    ax.set_ylabel("Mean pi_b")
    ax.set_title("Mean pi_b(a_clinician | s) per action")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bp_per_action.png"), dpi=200)
    plt.close()

    # per-action OPE quality: mean pi_b(a_clin|s) and frac<1%
    action_ids_ope, mean_probs_ope, frac_low_ope, n_steps = [], [], [], []
    for a in range(n_actions):
        sub = df[df["clin_action"] == a]
        if len(sub) == 0:
            continue
        action_ids_ope.append(a)
        mean_probs_ope.append(sub["pi_b_clin"].mean())
        frac_low_ope.append((sub["pi_b_clin"] < 0.01).mean())
        n_steps.append(len(sub))

    x_act = np.arange(len(action_ids_ope))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    bars = axes[0].bar(x_act, mean_probs_ope, color="steelblue", edgecolor="k")
    axes[0].axhline(1 / n_actions, color="red", linestyle="--", label=f"uniform={1/n_actions:.3f}")
    axes[0].set_ylabel("Mean pi_b(a_clin | s)")
    axes[0].set_title("Behavior policy quality per action (OPE perspective)")
    axes[0].legend()
    for i, (bar, n) in enumerate(zip(bars, n_steps)):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f"n={n//1000}k" if n >= 1000 else f"n={n}",
                     ha="center", va="bottom", fontsize=6, rotation=90)

    axes[1].bar(x_act, [f * 100 for f in frac_low_ope], color="tomato", edgecolor="k")
    axes[1].axhline(5, color="orange", linestyle="--", label="5% threshold")
    axes[1].set_ylabel("% steps with pi_b < 0.01 (IS blowup risk)")
    axes[1].set_xlabel("Clinician action")
    axes[1].set_xticks(x_act)
    axes[1].set_xticklabels(action_ids_ope)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bp_ope_quality_per_action.png"), dpi=200)
    plt.close()

    # IS ratio contribution: log10(1/pi_b)
    log_ratio = np.log10(1.0 / (pi_b + 1e-12))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(log_ratio, bins=60, edgecolor="k")
    ax.axvline(1, color="orange", linestyle="--", label="ratio=10")
    ax.axvline(2, color="red", linestyle="--", label="ratio=100")
    ax.set_xlabel("log10(1 / pi_b(a_clin))  = log10 IS denominator")
    ax.set_ylabel("count")
    ax.set_title("Per-step IS ratio contribution (denominator)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bp_is_ratio.png"), dpi=200)
    plt.close()

    print(f"\nPlots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default="analysis_outputs/bp")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows from {args.csv}")

    print_summary(df)
    make_plots(df, args.output_dir)


if __name__ == "__main__":
    main()
