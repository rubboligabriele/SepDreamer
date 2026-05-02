import os
import argparse
import numpy as np
from collections import Counter, defaultdict


def load_episodes(eps_dir):
    episodes = {}
    for fname in os.listdir(eps_dir):
        if fname.endswith(".npz"):
            stay_id = fname[:-4]
            path = os.path.join(eps_dir, fname)
            data = np.load(path)
            episodes[stay_id] = {k: data[k] for k in data.files}
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--num-actions", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    print(f"Loading episodes from: {args.episodes_dir}")
    episodes = load_episodes(args.episodes_dir)
    print(f"Loaded episodes: {len(episodes)}")

    action_counter = Counter()
    reward_by_action = defaultdict(list)
    mortality_by_action = defaultdict(list)

    frac0_per_episode = []
    mortality_per_episode = []
    length_per_episode = []

    for stay_id, ep in episodes.items():
        actions = np.argmax(ep["action"], axis=-1).reshape(-1)
        rewards = ep["reward"].reshape(-1)
        mortality = ep["mortality"].reshape(-1)

        action_counter.update(actions.tolist())

        for a, r, m in zip(actions, rewards, mortality):
            reward_by_action[int(a)].append(float(r))
            mortality_by_action[int(a)].append(float(m))

        frac0 = float(np.mean(actions == 0))
        mort_ep = float(np.max(mortality))

        frac0_per_episode.append(frac0)
        mortality_per_episode.append(mort_ep)
        length_per_episode.append(len(actions))

    frac0_per_episode = np.array(frac0_per_episode)
    mortality_per_episode = np.array(mortality_per_episode)
    length_per_episode = np.array(length_per_episode)

    total_actions = sum(action_counter.values())

    print("\n" + "=" * 80)
    print("ACTION DISTRIBUTION")
    print("=" * 80)

    for a, c in action_counter.most_common(args.top_k):
        print(f"action {a:02d}: count={c:8d}  freq={100*c/total_actions:6.2f}%")

    print("\n" + "=" * 80)
    print("ACTION 0 EPISODE-LEVEL CHECK")
    print("=" * 80)

    print(f"mean frac action 0 per episode: {frac0_per_episode.mean():.4f}")
    print(f"median frac action 0 per episode: {np.median(frac0_per_episode):.4f}")
    print(f"episodes with frac0 > 0.80: {(frac0_per_episode > 0.8).sum()}")
    print(f"episodes with frac0 < 0.20: {(frac0_per_episode < 0.2).sum()}")

    for thr_name, mask in [
        ("frac0 > 0.80", frac0_per_episode > 0.8),
        ("frac0 > 0.50", frac0_per_episode > 0.5),
        ("frac0 < 0.20", frac0_per_episode < 0.2),
        ("frac0 < 0.05", frac0_per_episode < 0.05),
    ]:
        if mask.sum() > 0:
            print(
                f"{thr_name:12s}: n={mask.sum():5d}, "
                f"mortality={mortality_per_episode[mask].mean():.4f}, "
                f"mean_len={length_per_episode[mask].mean():.2f}"
            )

    corr = np.corrcoef(frac0_per_episode, mortality_per_episode)[0, 1]
    print(f"corr(frac_action0, mortality): {corr:.4f}")

    print("\n" + "=" * 80)
    print("REWARD AND MORTALITY BY ACTION")
    print("=" * 80)

    print(
        f"{'action':>6} {'count':>10} {'freq%':>8} "
        f"{'reward_mean':>14} {'reward_std':>12} {'mort_mean':>12}"
    )

    for a in range(args.num_actions):
        rs = np.array(reward_by_action[a], dtype=np.float32)
        ms = np.array(mortality_by_action[a], dtype=np.float32)
        c = len(rs)

        if c == 0:
            print(f"{a:6d} {0:10d} {0.0:8.2f} {'nan':>14} {'nan':>12} {'nan':>12}")
            continue

        print(
            f"{a:6d} {c:10d} {100*c/total_actions:8.2f} "
            f"{rs.mean():14.6f} {rs.std():12.6f} {ms.mean():12.6f}"
        )

    print("\n" + "=" * 80)
    print("TERMINAL REWARD CHECK")
    print("=" * 80)

    terminal_rewards = []
    terminal_mortality = []

    for ep in episodes.values():
        if "is_terminal" not in ep:
            continue

        term = ep["is_terminal"].reshape(-1).astype(bool)
        rewards = ep["reward"].reshape(-1)
        mortality = ep["mortality"].reshape(-1)

        if term.any():
            terminal_rewards.extend(rewards[term].tolist())
            terminal_mortality.extend(mortality[term].tolist())

    terminal_rewards = np.array(terminal_rewards, dtype=np.float32)
    terminal_mortality = np.array(terminal_mortality, dtype=np.float32)

    if len(terminal_rewards) > 0:
        death_mask = terminal_mortality == 1
        surv_mask = terminal_mortality == 0

        print(f"num terminal rewards: {len(terminal_rewards)}")
        print(f"terminal reward mean: {terminal_rewards.mean():.6f}")
        print(f"terminal reward min/max: {terminal_rewards.min():.6f} / {terminal_rewards.max():.6f}")

        if death_mask.any():
            print(f"death terminal reward mean: {terminal_rewards[death_mask].mean():.6f}")
        if surv_mask.any():
            print(f"survival terminal reward mean: {terminal_rewards[surv_mask].mean():.6f}")
    else:
        print("No terminal rewards found.")


if __name__ == "__main__":
    main()