import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def summarize_npz(npz_path):
    data = np.load(npz_path)

    print("\n" + "=" * 80)
    print("NPZ SUMMARY")
    print("=" * 80)

    for key in data.files:
        arr = data[key]
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")

    phys = data["phys"]
    mort = data["mort"]

    print("\n[phys return stats]")
    print(f"mean   : {phys.mean():.6f}")
    print(f"std    : {phys.std():.6f}")
    print(f"min    : {phys.min():.6f}")
    print(f"max    : {phys.max():.6f}")
    print(f"median : {np.median(phys):.6f}")

    print("\n[mortality stats]")
    print(f"mean mortality: {mort.mean():.6f} ({100*mort.mean():.2f}%)")

    if "value" in data.files:
        value = data["value"]
        print("\n[value stats]")
        print(f"mean   : {value.mean():.6f}")
        print(f"std    : {value.std():.6f}")
        print(f"min    : {value.min():.6f}")
        print(f"max    : {value.max():.6f}")
        print(f"median : {np.median(value):.6f}")


def binned_curve(x, y, num_bins=20):
    q01, q99 = np.quantile(x, [0.01, 0.99])
    mask = (x >= q01) & (x <= q99)
    x = x[mask]
    y = y[mask]

    bins = np.linspace(q01, q99, num_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    inds = np.digitize(x, bins) - 1

    means = np.full(num_bins, np.nan)
    counts = np.zeros(num_bins, dtype=int)

    for i in range(num_bins):
        m = inds == i
        if m.sum() > 0:
            means[i] = y[m].mean()
            counts[i] = m.sum()

    valid = ~np.isnan(means)
    return centers[valid], means[valid], counts[valid]


def plot_npz_curves(npz_path, outdir):
    data = np.load(npz_path)
    phys = data["phys"]
    mort = data["mort"]

    os.makedirs(outdir, exist_ok=True)

    # Mortality vs physician return
    x, y, c = binned_curve(phys, mort, num_bins=20)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel("Physician episode return")
    plt.ylabel("Mortality")
    plt.title("Mortality vs Physician Episode Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mortality_vs_phys_return.png"), dpi=150)
    plt.close()

    if "value" in data.files:
        value = data["value"]
        x, y, c = binned_curve(value, mort, num_bins=20)
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, marker="o")
        plt.xlabel("Critic value")
        plt.ylabel("Mortality")
        plt.title("Mortality vs Critic Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "mortality_vs_value.png"), dpi=150)
        plt.close()


def summarize_csv(csv_path):
    df = pd.read_csv(csv_path)

    print("\n" + "=" * 80)
    print("CSV SUMMARY")
    print("=" * 80)
    print(df.head())
    print("\ncolumns:", list(df.columns))
    print(f"num rows: {len(df)}")

    if "mortality" in df.columns:
        print(f"\nmean mortality: {df['mortality'].mean():.6f} ({100*df['mortality'].mean():.2f}%)")

    if {"phys_action", "ai_action"}.issubset(df.columns):
        mismatch = (df["phys_action"] != df["ai_action"]).mean()
        print(f"\naction mismatch rate: {mismatch:.6f} ({100*mismatch:.2f}%)")

        print("\n[phys action distribution]")
        print(df["phys_action"].value_counts(normalize=True).sort_index())

        print("\n[ai action distribution]")
        print(df["ai_action"].value_counts(normalize=True).sort_index())

        if "mortality" in df.columns:
            print("\n[mortality by physician action]")
            print(df.groupby("phys_action")["mortality"].mean().sort_index())

            print("\n[mortality by AI action]")
            print(df.groupby("ai_action")["mortality"].mean().sort_index())


def plot_action_distributions(csv_path, outdir):
    df = pd.read_csv(csv_path)
    os.makedirs(outdir, exist_ok=True)

    if not {"phys_action", "ai_action"}.issubset(df.columns):
        return

    phys_counts = df["phys_action"].value_counts(normalize=True).sort_index()
    ai_counts = df["ai_action"].value_counts(normalize=True).sort_index()

    all_actions = sorted(set(phys_counts.index).union(set(ai_counts.index)))
    phys_vals = [phys_counts.get(a, 0.0) for a in all_actions]
    ai_vals = [ai_counts.get(a, 0.0) for a in all_actions]

    x = np.arange(len(all_actions))
    width = 0.4

    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, phys_vals, width=width, label="Physician")
    plt.bar(x + width / 2, ai_vals, width=width, label="AI")
    plt.xticks(x, all_actions, rotation=90)
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.title("Action distribution: Physician vs AI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "action_distribution_comparison.png"), dpi=150)
    plt.close()

    if "mortality" in df.columns:
        mort_ai = df.groupby("ai_action")["mortality"].mean()
        mort_phys = df.groupby("phys_action")["mortality"].mean()

        mort_ai_vals = [mort_ai.get(a, np.nan) for a in all_actions]
        mort_phys_vals = [mort_phys.get(a, np.nan) for a in all_actions]

        plt.figure(figsize=(10, 4))
        plt.plot(all_actions, mort_phys_vals, marker="o", label="Physician action mortality")
        plt.plot(all_actions, mort_ai_vals, marker="o", label="AI action mortality")
        plt.xlabel("Action")
        plt.ylabel("Mortality")
        plt.title("Mortality by action")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "mortality_by_action.png"), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="analysis_outputs")
    args = parser.parse_args()

    summarize_npz(args.npz)
    summarize_csv(args.csv)
    plot_npz_curves(args.npz, args.outdir)
    plot_action_distributions(args.csv, args.outdir)

    print("\nSaved plots to:", args.outdir)


if __name__ == "__main__":
    main()