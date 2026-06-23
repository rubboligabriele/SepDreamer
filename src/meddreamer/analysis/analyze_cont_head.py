"""
Analyze cont_head behavior: action sensitivity, temporal evolution, calibration.

Usage:
    python -u -m src.meddreamer.analysis.analyze_cont_head \
        --configs defaults eval \
        --ckptdir /path/to/wm/checkpoints \
        --ckptepoch 10000 \
        --output-dir analysis_outputs/cont_head \
        --max-episodes 2000
"""
import os
import argparse
import pathlib
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import src.meddreamer.utils.tools as tools
from src.meddreamer.dreamer import MedDreamer


CONTEXT_STEPS = 5
PROBE_STEPS = [4, 9, 14, 19, 24, 29]  # absolute timesteps to probe (relative to start)


def make_agent(config):
    eps_dir = os.path.join(config.datadir, config.dataset, "episodes")
    cache_root = os.path.dirname(eps_dir)
    split_path = os.path.join(cache_root, f"splits_seed{config.seed}.pkl")
    with open(split_path, "rb") as f:
        splits = pickle.load(f)
    test_cache = os.path.join(cache_root, f"test_eps_cache_seed{config.seed}.pkl")
    episodes = tools.load_split_episodes(eps_dir, splits["test"], cache_path=test_cache)

    logdir = pathlib.Path("tmp_cont_head_analysis")
    logdir.mkdir(exist_ok=True)
    logger = tools.Logger(logdir)
    agent = MedDreamer(config, logger, logdir, None, episodes).to(config.device)
    tools.load_model(agent, "wm", config.ckptdir, config.ckptepoch, config.device)
    agent._wm.eval()
    return agent, episodes


def collect(agent, episodes, max_episodes):
    """Collect per-step cont_prob and per-step action sensitivity for each episode."""
    num_actions = agent._config.num_actions
    cont_type = agent._config.cont_type

    # temporal evolution: list of (t, cont_prob, mortality)
    temporal = []

    # action sensitivity: list of (t, action_taken, cont_by_action[25], mortality)
    sensitivity = []

    count = 0
    total = min(max_episodes, len(episodes)) if max_episodes else len(episodes)

    with torch.no_grad():
        for stay_id, data in tqdm(episodes.items(), total=total, desc="Collecting", unit="ep"):
            if max_episodes and count >= max_episodes:
                break

            data = agent._expand_episode(data)
            B, T, _ = data["features"].shape
            if T <= CONTEXT_STEPS:
                continue

            data = agent._wm.preprocess(data)
            features = tools.bt_flatten(data["features"])
            delta = tools.bt_flatten(data["delta"]) if agent._config.fm["use_fm"] else None
            embed = tools.bt_unflatten(agent._wm.encoder(features, delta), B, T)

            phys_action = data["action"]
            is_first = data["is_first"]
            mortality = float(data["mortality"][0, 0].item())

            states, _ = agent._wm.dynamics.observe(
                embed[:, :CONTEXT_STEPS], phys_action[:, :CONTEXT_STEPS], is_first[:, :CONTEXT_STEPS]
            )
            # roll forward step by step keeping posterior states
            full_states, _ = agent._wm.dynamics.observe(embed, phys_action, is_first)
            feat_all = agent._wm.dynamics.get_feat(full_states)  # (1, T, D)

            # temporal evolution: cont_prob at every real step
            for t in range(T):
                feat_t = feat_all[:, t]
                if cont_type == "mort3":
                    cont_prob = float(agent._wm.heads["cont"](feat_t).probs[..., 2].squeeze().item())
                else:
                    cont_prob = float(agent._wm.heads["cont"](feat_t).mean.squeeze().item())
                temporal.append((t, cont_prob, mortality))

            # action sensitivity: probe at PROBE_STEPS
            for t in PROBE_STEPS:
                if t >= T:
                    continue
                state_t = {k: v[:, t] for k, v in full_states.items()}
                action_taken = int(torch.argmax(phys_action[0, t]).item())
                cont_by_action = []
                for a in range(num_actions):
                    action_a = torch.zeros((1, num_actions), device=agent._config.device)
                    action_a[:, a] = 1.0
                    next_state = agent._wm.dynamics.img_step(state_t, action_a, sample=False)
                    feat_next = agent._wm.dynamics.get_feat(next_state)
                    if cont_type == "mort3":
                        cp = float(agent._wm.heads["cont"](feat_next).probs[..., 2].squeeze().item())
                    else:
                        cp = float(agent._wm.heads["cont"](feat_next).mean.squeeze().item())
                    cont_by_action.append(cp)
                sensitivity.append((t, action_taken, np.array(cont_by_action, dtype=np.float32), mortality))

            count += 1

    return temporal, sensitivity


def plot_temporal_evolution(temporal, output_dir):
    """cont_prob(s_t) mean over time, split by mortality."""
    data = np.array([(t, cp, m) for t, cp, m in temporal], dtype=np.float32)
    max_t = int(data[:, 0].max()) + 1

    alive_mean, dead_mean, alive_sem, dead_sem = [], [], [], []
    ts = []
    for t in range(max_t):
        mask = data[:, 0] == t
        if mask.sum() < 5:
            continue
        alive = data[mask & (data[:, 2] == 0), 1]
        dead = data[mask & (data[:, 2] == 1), 1]
        if len(alive) < 2 or len(dead) < 2:
            continue
        alive_mean.append(alive.mean())
        dead_mean.append(dead.mean())
        alive_sem.append(alive.std() / np.sqrt(len(alive)))
        dead_sem.append(dead.std() / np.sqrt(len(dead)))
        ts.append(t)

    ts = np.array(ts)
    alive_mean = np.array(alive_mean)
    dead_mean = np.array(dead_mean)
    alive_sem = np.array(alive_sem)
    dead_sem = np.array(dead_sem)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts, alive_mean, label="Survived", color="steelblue")
    ax.fill_between(ts, alive_mean - alive_sem, alive_mean + alive_sem, alpha=0.3, color="steelblue")
    ax.plot(ts, dead_mean, label="Died", color="tomato")
    ax.fill_between(ts, dead_mean - dead_sem, dead_mean + dead_sem, alpha=0.3, color="tomato")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Step t", fontsize=13)
    ax.set_ylabel("cont_prob (mean ± SEM)", fontsize=13)
    ax.set_title("Cont head temporal evolution: survived vs died")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cont_temporal_evolution.png"), dpi=150)
    plt.close(fig)
    print("Saved cont_temporal_evolution.png")


def plot_cont_distribution(temporal, output_dir):
    """Distribution of cont_prob for died vs survived episodes."""
    data = np.array([(cp, m) for _, cp, m in temporal], dtype=np.float32)
    alive = data[data[:, 1] == 0, 0]
    dead = data[data[:, 1] == 1, 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 40)
    ax.hist(alive, bins=bins, alpha=0.6, label="Survived", color="steelblue", density=True)
    ax.hist(dead, bins=bins, alpha=0.6, label="Died", color="tomato", density=True)
    ax.set_xlabel("cont_prob", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("cont_prob distribution: survived vs died (all steps)")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cont_distribution.png"), dpi=150)
    plt.close(fig)
    print("Saved cont_distribution.png")



def plot_best_action_by_step(sensitivity, num_actions, output_dir):
    """For each probe step t: distribution of which action maximizes cont."""
    steps = sorted(set(t for t, _, _, _ in sensitivity))
    fig, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 4), sharey=True)
    if len(steps) == 1:
        axes = [axes]

    for ax, t in zip(axes, steps):
        rows = [(cont, m) for st, _, cont, m in sensitivity if st == t]
        if not rows:
            continue
        best_actions = [int(np.argmax(cont)) for cont, _ in rows]
        counts = np.bincount(best_actions, minlength=num_actions)
        ax.bar(range(num_actions), counts / max(counts.sum(), 1))
        ax.set_title(f"t={t}")
        ax.set_xlabel("Action")
        if ax == axes[0]:
            ax.set_ylabel("Fraction best cont")

    fig.suptitle("Best cont action distribution by timestep")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cont_best_action_by_step.png"), dpi=150)
    plt.close(fig)
    print("Saved cont_best_action_by_step.png")


def build_config(config_names, remaining):
    import sys
    import pathlib
    from ruamel.yaml import YAML

    yaml = YAML(typ="rt")
    configs_path = pathlib.Path(__file__).parent.parent / "configs.yaml"
    all_configs = yaml.load(configs_path.read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *config_names] if config_names else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, all_configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    return parser.parse_args(remaining)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["defaults"])
    parser.add_argument("--ckptdir", required=True)
    parser.add_argument("--ckptepoch", type=int, default=10000)
    parser.add_argument("--output-dir", default="analysis_outputs/cont_head")
    parser.add_argument("--max-episodes", type=int, default=None)
    args, remaining = parser.parse_known_args()

    config = build_config(args.configs, remaining)
    config.ckptdir = args.ckptdir
    config.ckptepoch = args.ckptepoch

    os.makedirs(args.output_dir, exist_ok=True)

    agent, episodes = make_agent(config)
    temporal, sensitivity = collect(agent, episodes, args.max_episodes)

    print(f"\nCollected {len(temporal)} step-records from {len(set(t for t,_,_ in temporal))} unique timesteps")
    print(f"Collected {len(sensitivity)} sensitivity records")

    plot_temporal_evolution(temporal, args.output_dir)
    plot_cont_distribution(temporal, args.output_dir)
    plot_action_heatmap(sensitivity, agent._config.num_actions, args.output_dir)
    plot_best_action_by_step(sensitivity, agent._config.num_actions, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
