import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def action_to_bins(action_id, n_bins=5):
    action_id = int(action_id)
    fluid_bin = action_id // n_bins
    vaso_bin = action_id % n_bins
    return fluid_bin, vaso_bin


def confusion_matrix_5x5(y_true, y_pred, n_bins=5):
    cm = np.zeros((n_bins, n_bins), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < n_bins and 0 <= int(p) < n_bins:
            cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", origin="lower", cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("AI action bin")
    ax.set_ylabel("Physician action bin")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_joint_action_heatmap(df, output_dir, n_bins=5):
    """Replicates Figure 2 from the medDreamer paper: joint fluid x vaso distribution heatmap."""
    df = add_fluid_vaso_columns(df.copy(), n_bins=n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = ["Clinician", "MedDreamer"]
    fluid_cols = ["phys_fluid", "ai_fluid"]
    vaso_cols = ["phys_vaso", "ai_vaso"]
    cmaps = ["Blues", "Purples"]

    for ax, title, fcol, vcol, cmap in zip(axes, titles, fluid_cols, vaso_cols, cmaps):
        heatmap = np.zeros((n_bins, n_bins))
        for f, v in zip(df[fcol], df[vcol]):
            heatmap[int(v), int(f)] += 1  # row=vaso, col=fluid

        im = ax.imshow(
            heatmap,
            interpolation="nearest",
            vmin=0,
            vmax=heatmap.max(),
            cmap=cmap,
            origin="lower",
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("IV Fluid", fontsize=11)
        ax.set_ylabel("Vasopressor", fontsize=11)
        ax.set_xticks(range(n_bins))
        ax.set_yticks(range(n_bins))
        fig.colorbar(im, ax=ax)

    plt.suptitle("Joint Action Distribution: IV Fluid × Vasopressor", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "joint_action_heatmap.png"), dpi=200)
    plt.close()


def add_fluid_vaso_columns(df, n_bins=5):
    phys = df["phys_action"].astype(int).to_numpy()
    ai = df["ai_action"].astype(int).to_numpy()

    df["phys_fluid"] = phys // n_bins
    df["phys_vaso"] = phys % n_bins
    df["ai_fluid"] = ai // n_bins
    df["ai_vaso"] = ai % n_bins

    return df


def summarize_array(name, x):
    print(f"\n[{name} stats]")
    print(f"mean   : {np.mean(x):.6f}")
    print(f"std    : {np.std(x):.6f}")
    print(f"min    : {np.min(x):.6f}")
    print(f"max    : {np.max(x):.6f}")
    print(f"median : {np.median(x):.6f}")


def analyze_npz(npz_path):
    data = np.load(npz_path)

    print("\n" + "=" * 80)
    print("NPZ SUMMARY")
    print("=" * 80)

    for k in data.files:
        print(f"{k}: shape={data[k].shape}, dtype={data[k].dtype}")

    if "phys" in data:
        summarize_array("phys return", data["phys"])

    if "mort" in data:
        mort = data["mort"]
        print("\n[mortality stats]")
        print(f"mean mortality: {np.mean(mort):.6f} ({100 * np.mean(mort):.2f}%)")

    if "value" in data:
        summarize_array("value", data["value"])

    return data


def plot_action_distribution(df, output_dir):
    phys_dist = df["phys_action"].value_counts(normalize=True).sort_index()
    ai_dist = df["ai_action"].value_counts(normalize=True).sort_index()

    actions = sorted(set(phys_dist.index).union(set(ai_dist.index)))
    phys_vals = [phys_dist.get(a, 0.0) for a in actions]
    ai_vals = [ai_dist.get(a, 0.0) for a in actions]

    x = np.arange(len(actions))
    width = 0.38

    plt.figure(figsize=(14, 5))
    plt.bar(x - width / 2, phys_vals, width, label="Physician")
    plt.bar(x + width / 2, ai_vals, width, label="AI")
    plt.xticks(x, actions, rotation=90)
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.title("Action distribution: Physician vs AI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_distribution_physician_vs_ai.png"), dpi=200)
    plt.close()


def plot_mortality_by_action(df, output_dir):
    phys_mort = df.groupby("phys_action")["mortality"].mean().sort_index()
    ai_mort = df.groupby("ai_action")["mortality"].mean().sort_index()

    actions = sorted(set(phys_mort.index).union(set(ai_mort.index)))

    plt.figure(figsize=(14, 5))
    plt.plot(actions, [phys_mort.get(a, np.nan) for a in actions], marker="o", label="Physician action mortality")
    plt.plot(actions, [ai_mort.get(a, np.nan) for a in actions], marker="o", label="AI action mortality")
    plt.xlabel("Action")
    plt.ylabel("Mortality")
    plt.title("Mortality by action")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mortality_by_action.png"), dpi=200)
    plt.close()


def plot_binned_curve(x, y, title, xlabel, ylabel, out_path, bins=20):
    x = np.asarray(x)
    y = np.asarray(y)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return

    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)

    xs = []
    ys = []

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)

        if mask.sum() > 0:
            xs.append(float(np.mean(x[mask])))
            ys.append(float(np.mean(y[mask])))

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def analyze_separate_fluid_vaso_confusions(df, output_dir, n_bins=5):
    df = add_fluid_vaso_columns(df.copy(), n_bins=n_bins)

    fluid_cm = confusion_matrix_5x5(df["phys_fluid"], df["ai_fluid"], n_bins=n_bins)
    vaso_cm = confusion_matrix_5x5(df["phys_vaso"], df["ai_vaso"], n_bins=n_bins)

    print("\n" + "=" * 80)
    print("SEPARATE FLUID/VASO AGREEMENT")
    print("=" * 80)

    joint_acc = np.mean(df["phys_action"] == df["ai_action"])
    fluid_acc = np.mean(df["phys_fluid"] == df["ai_fluid"])
    vaso_acc = np.mean(df["phys_vaso"] == df["ai_vaso"])

    print(f"joint action agreement : {100 * joint_acc:.2f}%")
    print(f"fluid bin agreement    : {100 * fluid_acc:.2f}%")
    print(f"vaso bin agreement     : {100 * vaso_acc:.2f}%")

    print("\n[Physician fluid distribution]")
    print(df["phys_fluid"].value_counts(normalize=True).sort_index())

    print("\n[AI fluid distribution]")
    print(df["ai_fluid"].value_counts(normalize=True).sort_index())

    print("\n[Physician vaso distribution]")
    print(df["phys_vaso"].value_counts(normalize=True).sort_index())

    print("\n[AI vaso distribution]")
    print(df["ai_vaso"].value_counts(normalize=True).sort_index())

    print("\n[Fluid confusion matrix: rows=physician, cols=AI]")
    print(fluid_cm)

    print("\n[Vaso confusion matrix: rows=physician, cols=AI]")
    print(vaso_cm)

    plot_confusion_matrix(
        fluid_cm,
        "IV Fluids: Physician vs AI",
        os.path.join(output_dir, "confusion_iv_fluids.png"),
    )

    plot_confusion_matrix(
        vaso_cm,
        "Vasopressors: Physician vs AI",
        os.path.join(output_dir, "confusion_vasopressors.png"),
    )

    plot_joint_action_heatmap(df, output_dir, n_bins=n_bins)


def analyze_csv(csv_path, output_dir, n_bins=5):
    df = pd.read_csv(csv_path)

    print("\n" + "=" * 80)
    print("CSV SUMMARY")
    print("=" * 80)

    print(df.head())
    print(f"\ncolumns: {list(df.columns)}")
    print(f"num rows: {len(df)}")

    required = ["mortality", "phys_action", "ai_action"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    print(f"\nmean mortality: {df['mortality'].mean():.6f} ({100 * df['mortality'].mean():.2f}%)")

    mismatch_rate = np.mean(df["phys_action"] != df["ai_action"])
    print(f"\naction mismatch rate: {mismatch_rate:.6f} ({100 * mismatch_rate:.2f}%)")

    print("\n[phys action distribution]")
    print(df["phys_action"].value_counts(normalize=True).sort_index())

    print("\n[ai action distribution]")
    print(df["ai_action"].value_counts(normalize=True).sort_index())

    print("\n[mortality by physician action]")
    print(df.groupby("phys_action")["mortality"].mean().sort_index())

    print("\n[mortality by AI action]")
    print(df.groupby("ai_action")["mortality"].mean().sort_index())

    plot_action_distribution(df, output_dir)
    plot_mortality_by_action(df, output_dir)
    analyze_separate_fluid_vaso_confusions(df, output_dir, n_bins=n_bins)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default="analysis_outputs")
    parser.add_argument("--num-bins", type=int, default=5)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    npz_data = analyze_npz(args.npz)
    df = analyze_csv(args.csv, args.output_dir, n_bins=args.num_bins)

    if "phys" in npz_data and "mort" in npz_data:
        plot_binned_curve(
            x=npz_data["phys"],
            y=npz_data["mort"],
            title="Mortality vs Physician Episode Return",
            xlabel="Physician episode return",
            ylabel="Mortality",
            out_path=os.path.join(args.output_dir, "mortality_vs_physician_episode_return.png"),
            bins=20,
        )

    if "value" in npz_data and "mort" in npz_data:
        plot_binned_curve(
            x=npz_data["value"],
            y=npz_data["mort"],
            title="Mortality vs Critic Value",
            xlabel="Critic value",
            ylabel="Mortality",
            out_path=os.path.join(args.output_dir, "mortality_vs_critic_value.png"),
            bins=20,
        )

    print(f"\nSaved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()