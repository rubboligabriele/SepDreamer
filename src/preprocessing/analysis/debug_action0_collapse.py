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


def action_to_bins(action_id, n_bins=5):
    fluid_bin = action_id // n_bins
    vaso_bin = action_id % n_bins
    return fluid_bin, vaso_bin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--num-actions", type=int, default=25)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    print(f"Loading episodes from: {args.episodes_dir}")
    episodes = load_episodes(args.episodes_dir)
    print(f"Loaded episodes: {len(episodes)}")

    action_counter = Counter()
    fluid_counter = Counter()
    vaso_counter = Counter()

    reward_by_action = defaultdict(list)
    mortality_by_action = defaultdict(list)

    frac0_per_episode = []
    frac0_excluding_first_per_episode = []
    mortality_per_episode = []
    length_per_episode = []

    bad_onehot = 0
    bad_action_range = 0
    bad_lengths = 0
    nan_inf_eps = 0
    first_action_not_zero = 0

    terminal_rewards = []
    terminal_mortality = []

    first_step_action_counter = Counter()
    nonfirst_action_counter = Counter()

    for stay_id, ep in episodes.items():
        required = ["action", "reward", "mortality", "is_terminal"]
        missing = [k for k in required if k not in ep]
        if missing:
            print(f"[BAD] stay {stay_id}: missing keys {missing}")
            continue

        action_arr = ep["action"]
        rewards = ep["reward"].reshape(-1)
        mortality = ep["mortality"].reshape(-1)
        is_terminal = ep["is_terminal"].reshape(-1)

        T = len(rewards)

        if action_arr.shape[0] != T or len(mortality) != T or len(is_terminal) != T:
            bad_lengths += 1
            print(
                f"[BAD LENGTH] stay {stay_id}: "
                f"action={action_arr.shape}, reward={rewards.shape}, "
                f"mortality={mortality.shape}, is_terminal={is_terminal.shape}"
            )
            continue

        if not np.isfinite(action_arr).all() or not np.isfinite(rewards).all() or not np.isfinite(mortality).all():
            nan_inf_eps += 1

        row_sums = action_arr.sum(axis=-1)
        if not np.allclose(row_sums, 1.0):
            bad_onehot += 1
            print(f"[BAD ONEHOT SUM] stay {stay_id}: min={row_sums.min()}, max={row_sums.max()}")

        if not np.all((action_arr == 0.0) | (action_arr == 1.0)):
            bad_onehot += 1
            print(f"[BAD ONEHOT VALUES] stay {stay_id}: non-binary action matrix")

        actions = np.argmax(action_arr, axis=-1).reshape(-1)

        if actions.min() < 0 or actions.max() >= args.num_actions:
            bad_action_range += 1
            print(f"[BAD ACTION RANGE] stay {stay_id}: min={actions.min()}, max={actions.max()}")

        if actions[0] != 0:
            first_action_not_zero += 1

        first_step_action_counter[int(actions[0])] += 1
        nonfirst_action_counter.update(actions[1:].tolist())
        action_counter.update(actions.tolist())

        for a in actions:
            fb, vb = action_to_bins(int(a), args.num_bins)
            fluid_counter[fb] += 1
            vaso_counter[vb] += 1

        for a, r, m in zip(actions, rewards, mortality):
            reward_by_action[int(a)].append(float(r))
            mortality_by_action[int(a)].append(float(m))

        frac0_per_episode.append(float(np.mean(actions == 0)))
        if len(actions) > 1:
            frac0_excluding_first_per_episode.append(float(np.mean(actions[1:] == 0)))
        else:
            frac0_excluding_first_per_episode.append(np.nan)

        mortality_per_episode.append(float(np.max(mortality)))
        length_per_episode.append(len(actions))

        term = is_terminal.astype(bool)
        if term.sum() != 1:
            print(f"[TERMINAL WARNING] stay {stay_id}: num terminal flags = {term.sum()}")

        if term.any():
            terminal_rewards.extend(rewards[term].tolist())
            terminal_mortality.extend(mortality[term].tolist())

    frac0_per_episode = np.array(frac0_per_episode)
    frac0_excluding_first_per_episode = np.array(frac0_excluding_first_per_episode)
    mortality_per_episode = np.array(mortality_per_episode)
    length_per_episode = np.array(length_per_episode)

    total_actions = sum(action_counter.values())

    print("\n" + "=" * 80)
    print("BASIC VALIDATION")
    print("=" * 80)
    print(f"bad onehot episodes: {bad_onehot}")
    print(f"bad action range episodes: {bad_action_range}")
    print(f"bad length episodes: {bad_lengths}")
    print(f"episodes with NaN/inf: {nan_inf_eps}")
    print(f"episodes where first action != 0: {first_action_not_zero}")
    print(f"total transitions/timesteps: {total_actions}")
    print(f"num unique actions used: {len(action_counter)} / {args.num_actions}")

    print("\n" + "=" * 80)
    print("ACTION DISTRIBUTION")
    print("=" * 80)
    for a, c in action_counter.most_common(args.top_k):
        fb, vb = action_to_bins(a, args.num_bins)
        print(
            f"action {a:02d} (fluid_bin={fb}, vaso_bin={vb}): "
            f"count={c:8d}  freq={100*c/total_actions:6.2f}%"
        )

    missing_actions = [a for a in range(args.num_actions) if action_counter[a] == 0]
    rare_actions = [a for a in range(args.num_actions) if 0 < action_counter[a] < 100]
    print(f"\nmissing actions: {missing_actions}")
    print(f"rare actions count < 100: {rare_actions}")

    print("\n" + "=" * 80)
    print("FLUID/VASO MARGINAL BIN DISTRIBUTION")
    print("=" * 80)
    print("Fluid bins:")
    for b in range(args.num_bins):
        c = fluid_counter[b]
        print(f"fluid_bin {b}: count={c:8d} freq={100*c/total_actions:6.2f}%")

    print("\nVaso bins:")
    for b in range(args.num_bins):
        c = vaso_counter[b]
        print(f"vaso_bin  {b}: count={c:8d} freq={100*c/total_actions:6.2f}%")

    print("\n" + "=" * 80)
    print("FIRST STEP / SHIFT CHECK")
    print("=" * 80)
    print("First-step action distribution:")
    for a, c in first_step_action_counter.most_common():
        print(f"action {a:02d}: count={c:8d} freq={100*c/sum(first_step_action_counter.values()):6.2f}%")

    print("\nNon-first action distribution top:")
    nonfirst_total = sum(nonfirst_action_counter.values())
    for a, c in nonfirst_action_counter.most_common(args.top_k):
        print(f"action {a:02d}: count={c:8d} freq={100*c/nonfirst_total:6.2f}%")

    print("\n" + "=" * 80)
    print("ACTION 0 EPISODE-LEVEL CHECK")
    print("=" * 80)
    print(f"global action 0 freq: {100*action_counter[0]/total_actions:.2f}%")
    print(f"global action 0 freq excluding first timestep: {100*nonfirst_action_counter[0]/nonfirst_total:.2f}%")
    print(f"mean frac action 0 per episode: {frac0_per_episode.mean():.4f}")
    print(f"median frac action 0 per episode: {np.median(frac0_per_episode):.4f}")
    print(f"mean frac action 0 excluding first: {np.nanmean(frac0_excluding_first_per_episode):.4f}")
    print(f"median frac action 0 excluding first: {np.nanmedian(frac0_excluding_first_per_episode):.4f}")
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

    if len(frac0_per_episode) > 1:
        corr = np.corrcoef(frac0_per_episode, mortality_per_episode)[0, 1]
        print(f"corr(frac_action0, mortality): {corr:.4f}")

    print("\n" + "=" * 80)
    print("REWARD AND MORTALITY BY ACTION")
    print("=" * 80)
    print(
        f"{'action':>6} {'fluid':>6} {'vaso':>6} {'count':>10} {'freq%':>8} "
        f"{'reward_mean':>14} {'reward_std':>12} {'mort_mean':>12}"
    )

    for a in range(args.num_actions):
        rs = np.array(reward_by_action[a], dtype=np.float32)
        ms = np.array(mortality_by_action[a], dtype=np.float32)
        c = len(rs)
        fb, vb = action_to_bins(a, args.num_bins)

        if c == 0:
            print(f"{a:6d} {fb:6d} {vb:6d} {0:10d} {0.0:8.2f} {'nan':>14} {'nan':>12} {'nan':>12}")
            continue

        print(
            f"{a:6d} {fb:6d} {vb:6d} {c:10d} {100*c/total_actions:8.2f} "
            f"{rs.mean():14.6f} {rs.std():12.6f} {ms.mean():12.6f}"
        )

    print("\n" + "=" * 80)
    print("TERMINAL REWARD CHECK")
    print("=" * 80)

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
            print(f"death terminal reward unique: {np.unique(terminal_rewards[death_mask])[:20]}")

        if surv_mask.any():
            print(f"survival terminal reward mean: {terminal_rewards[surv_mask].mean():.6f}")
            print(f"survival terminal reward unique: {np.unique(terminal_rewards[surv_mask])[:20]}")

        bad_death = death_mask & (terminal_rewards >= 0)
        bad_surv = surv_mask & (terminal_rewards <= 0)
        print(f"bad death terminal reward >= 0: {bad_death.sum()}")
        print(f"bad survival terminal reward <= 0: {bad_surv.sum()}")
    else:
        print("No terminal rewards found.")

    print("\n" + "=" * 80)
    print("INTERPRETATION FLAGS")
    print("=" * 80)

    action0_freq = action_counter[0] / total_actions
    action0_nonfirst_freq = nonfirst_action_counter[0] / nonfirst_total

    if action0_freq > 0.70:
        print("RED FLAG: action 0 globally > 70%. Possible binning/alignment issue.")
    elif action0_freq > 0.50:
        print("WARNING: action 0 globally > 50%. Could be valid, but inspect bins.")
    else:
        print("OK: action 0 global frequency is not extreme.")

    if action0_nonfirst_freq > 0.70:
        print("RED FLAG: action 0 excluding first timestep > 70%. Not caused only by no-op t=0.")
    else:
        print("OK: action 0 excluding first timestep is not extreme.")

    if len(missing_actions) > 0:
        print("WARNING: some actions never appear. Check action binning if many are missing.")
    else:
        print("OK: all actions appear at least once.")

    if bad_onehot or bad_action_range or bad_lengths or nan_inf_eps:
        print("RED FLAG: dataset integrity issues found.")
    else:
        print("OK: no basic dataset integrity issues found.")


if __name__ == "__main__":
    main()