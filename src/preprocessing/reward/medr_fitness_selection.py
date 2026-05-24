from html import parser
import os
import json
import pickle
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm


CRITICAL_FEATURE_ALIASES = {
    "sofa": ["sofa"],
    "baseexcess": ["baseexcess", "base_excess", "arterial_be", "be"],
    "lactate": ["lactate"],
    "urineoutput": ["urineoutput", "urine_output", "output"],
    "mbp": ["mbp", "meanbp", "mean_bp"],
    "heartrate": ["heartrate", "heart_rate", "hr"],
}


DEFAULT_CANDIDATES = [
    {
        "name": "baseline_low_cost",
        "k_scale": 1.0,
        "sigma_scale": 1.0,
        "confidence_tau": 6.0,
        "half_life": 48.0,
        "survival_weight": 0.7,
        "confidence_weight": 0.3,
        "action_cost_scale": 0.005,
        "fluid_cost_weight": 1.0,
        "vaso_cost_weight": 1.0,
        "potential_diff_scale": 20.0,
        "use_time_decay": False,
    },
    {
        "name": "paper_like",
        "k_scale": 1.0,
        "sigma_scale": 1.0,
        "confidence_tau": 6.0,
        "half_life": 48.0,
        "survival_weight": 0.7,
        "confidence_weight": 0.3,
        "action_cost_scale": 0.02,
        "fluid_cost_weight": 1.0,
        "vaso_cost_weight": 1.0,
        "potential_diff_scale": 20.0,
        "use_time_decay": False,
    },
    {
        "name": "very_low_cost",
        "k_scale": 1.0,
        "sigma_scale": 1.0,
        "confidence_tau": 12.0,
        "half_life": 72.0,
        "survival_weight": 0.8,
        "confidence_weight": 0.2,
        "action_cost_scale": 0.001,
        "fluid_cost_weight": 1.0,
        "vaso_cost_weight": 1.0,
        "potential_diff_scale": 20.0,
        "use_time_decay": False,
    },
]

NORMAL_INTERVAL = (0.4, 0.6)
NORMAL_IQR = NORMAL_INTERVAL[1] - NORMAL_INTERVAL[0]


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def load_episodes(episodes_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    episodes = {}

    files = [f for f in os.listdir(episodes_dir) if f.endswith(".npz")]

    for fname in tqdm(files, desc="Loading episodes"):
        path = os.path.join(episodes_dir, fname)
        data = np.load(path, allow_pickle=True)
        episodes[fname[:-4]] = {k: data[k] for k in data.files}

    return episodes


def subsample_episodes(
    episodes: Dict[str, Dict[str, np.ndarray]],
    max_episodes: int | None,
    seed: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    if max_episodes is None or max_episodes <= 0 or max_episodes >= len(episodes):
        return episodes

    rng = np.random.default_rng(seed)
    keys = np.array(list(episodes.keys()))
    chosen = rng.choice(keys, size=max_episodes, replace=False)

    return {k: episodes[k] for k in chosen}


def load_feature_cols(dataset_dir: str) -> List[str]:
    path = os.path.join(dataset_dir, "column_config.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing column_config.pkl: {path}")

    with open(path, "rb") as f:
        cfg = pickle.load(f)

    return list(cfg["feature_cols"])


def find_feature_indices(feature_cols: List[str]) -> Dict[str, int]:
    out = {}
    lower_cols = [c.lower() for c in feature_cols]

    print("\nCritical feature mapping:")

    for canonical_name, aliases in CRITICAL_FEATURE_ALIASES.items():
        matches = []

        for i, col in enumerate(lower_cols):
            for alias in aliases:
                if alias in col:
                    matches.append(i)
                    break

        if not matches:
            print(f"[WARNING] Could not find feature for {canonical_name}")
            continue

        out[canonical_name] = matches[0]
        print(f"{canonical_name:12s} -> {feature_cols[matches[0]]}")

    return out


def action_id_to_bins(action_id: int, n_bins: int = 5) -> Tuple[int, int]:
    fluid_bin = action_id // n_bins
    vaso_bin = action_id % n_bins
    return fluid_bin, vaso_bin


def action_magnitude(action_id: int, n_bins: int = 5) -> float:
    fluid_bin, vaso_bin = action_id_to_bins(action_id, n_bins)
    return 0.5 * (
        fluid_bin / float(n_bins - 1) +
        vaso_bin / float(n_bins - 1)
    )


def candidate_action_cost(
    action_id: int,
    candidate: Dict[str, Any],
    n_bins: int = 5,
) -> float:
    fluid_bin, vaso_bin = action_id_to_bins(action_id, n_bins)

    max_bin = float(n_bins - 1)
    fluid_norm = fluid_bin / max_bin
    vaso_norm = vaso_bin / max_bin

    cost_scale = float(candidate.get("action_cost_scale", 0.02))
    fluid_w = float(candidate.get("fluid_cost_weight", 1.0))
    vaso_w = float(candidate.get("vaso_cost_weight", 1.0))

    return float(cost_scale * (fluid_w * fluid_norm + vaso_w * vaso_norm))


def candidate_potential(
    x: np.ndarray,
    d: np.ndarray,
    feature_indices: Dict[str, int],
    candidate: Dict[str, Any],
    t: int,
) -> float:
    survival_scores = []
    confidence_scores = []

    k_scale = float(candidate.get("k_scale", 1.0))
    sigma_scale = float(candidate.get("sigma_scale", 1.0))
    tau = float(candidate.get("confidence_tau", 6.0))
    half_life = float(candidate.get("half_life", 48.0))

    for name, idx in feature_indices.items():
        val = float(x[idx])
        delta_t = float(d[idx])

        if not np.isfinite(val):
            continue

        # Your episodes use DataNormalization, not medR min-max.
        # So we squash normalized values to pseudo-[0,1].
        z = float(sigmoid(val))

        if name in ["sofa", "lactate"]:
            # Lower is better.
            surv = float(np.exp(-k_scale * z))

        elif name == "urineoutput":
            # Higher is better.
            surv = float(np.exp(-k_scale * max(0.0, 0.5 - z)))

        else:
            # Goldilocks features around pseudo-normal center.
            sigma = max(1e-6, 0.15 * sigma_scale)
            surv = float(np.exp(-0.5 * ((z - 0.5) / sigma) ** 2))

        if not np.isfinite(delta_t) or delta_t < 0:
            delta_t = 0.0

        conf = float(np.exp(-delta_t / tau))

        survival_scores.append(surv * conf)
        confidence_scores.append(conf)

    if len(survival_scores) == 0:
        base = 0.0
    else:
        survival_component = float(
            np.sum(survival_scores) / (np.sum(confidence_scores) + 1e-8)
        )
        confidence_component = float(np.mean(confidence_scores))

        survival_w = float(candidate.get("survival_weight", 0.7))
        confidence_w = float(candidate.get("confidence_weight", 0.3))

        weight_sum = survival_w + confidence_w
        if weight_sum <= 0:
            survival_w, confidence_w = 0.7, 0.3
            weight_sum = 1.0

        survival_w /= weight_sum
        confidence_w /= weight_sum

        base = survival_w * survival_component + confidence_w * confidence_component

    if bool(candidate.get("use_time_decay", False)):
        decay = 0.5 ** (float(t) / half_life)
    else:
        decay = 1.0

    return float(np.clip(base, 0.0, 1.0) * decay)


def compute_candidate_reward_sequence(
    ep: Dict[str, np.ndarray],
    feature_indices: Dict[str, int],
    candidate: Dict[str, Any],
    gamma: float = 0.99,
    n_bins: int = 5,
) -> np.ndarray:
    features = ep["features"].astype(np.float64)
    delta = ep["delta"].astype(np.float64)
    action_onehot = ep["action"].astype(np.float64)

    T = len(features)
    actions = np.argmax(action_onehot, axis=-1)

    rewards = np.zeros(T, dtype=np.float64)

    # reward[t] corresponds to transition t-1 -> t.
    # In your episode format, actions[t] is the shifted previous action,
    # i.e. actions[t] = a_{t-1}, the action that produced state s_t.
    for t in range(1, T):
        phi_prev = candidate_potential(
            x=features[t - 1],
            d=delta[t - 1],
            feature_indices=feature_indices,
            candidate=candidate,
            t=t - 1,
        )

        phi_cur = candidate_potential(
            x=features[t],
            d=delta[t],
            feature_indices=feature_indices,
            candidate=candidate,
            t=t,
        )

        cost = candidate_action_cost(
            action_id=int(actions[t]),
            candidate=candidate,
            n_bins=n_bins,
        )

        potential_diff_scale = float(candidate.get("potential_diff_scale", 20.0))

        potential_reward = potential_diff_scale * (gamma * phi_cur - phi_prev)

        # Same logic as the corrected training reward:
        # penalize intervention only when physiology did not improve.
        if phi_cur > phi_prev:
            cost = 0.0

        rewards[t] = potential_reward - cost

    return rewards


def homeostasis_score(
    x: np.ndarray,
    feature_indices: Dict[str, int],
    k: float = 10.0,
) -> float:
    scores = []

    for name, idx in feature_indices.items():
        val = float(x[idx])

        if not np.isfinite(val):
            continue

        z = float(sigmoid(val))

        if name in ["sofa", "lactate"]:
            h = float(sigmoid(k * (0.5 - z)))

        elif name == "urineoutput":
            h = float(sigmoid(k * (z - 0.5)))

        else:
            lo, hi = NORMAL_INTERVAL

            if lo <= z <= hi:
                h = 1.0
            else:
                dist = lo - z if z < lo else z - hi
                h = float(sigmoid(k * (0.5 - dist / NORMAL_IQR)))

        scores.append(h)

    if not scores:
        return np.nan

    return float(np.mean(scores))


def compute_episode_fitness_terms(
    ep: Dict[str, np.ndarray],
    feature_indices: Dict[str, int],
    candidate: Dict[str, Any],
    n_bins: int = 5,
    gamma: float = 0.99,
    sofa_eps: float = 2.0,
    alpha: float = 0.1,
    k_homeostasis: float = 10.0,
) -> Tuple[float, float, float, float]:
    reward = compute_candidate_reward_sequence(
        ep=ep,
        feature_indices=feature_indices,
        candidate=candidate,
        gamma=gamma,
        n_bins=n_bins,
    )

    mortality = ep["mortality"].reshape(-1).astype(np.float64)
    sofa = ep["sofa"].reshape(-1).astype(np.float64)
    features = ep["features"].astype(np.float64)
    delta = ep["delta"].astype(np.float64)
    action_onehot = ep["action"].astype(np.float64)

    T = len(reward)
    if T < 2:
        return np.nan, np.nan, np.nan, np.nan

    actions = np.argmax(action_onehot, axis=-1)

    cumulative_reward = float(np.sum(reward[1:]))

    survival = 1.0 - float(mortality[0])

    if np.isfinite(sofa[0]):
        sofa_base = sofa[0]
        sofa_stable = np.nanmean(np.abs(sofa - sofa_base) < sofa_eps)
    else:
        sofa_stable = 0.0

    if not np.isfinite(sofa_stable):
        sofa_stable = 0.0

    G_survival = survival + float(sofa_stable)

    crit_idx = list(feature_indices.values())
    if len(crit_idx) > 0:
        U_uncertainty = float(np.nanmean(delta[:, crit_idx]))
    else:
        U_uncertainty = np.nan

    efficiencies = []

    for t in range(1, T):
        h_prev = homeostasis_score(
            features[t - 1],
            feature_indices,
            k=k_homeostasis,
        )
        h_cur = homeostasis_score(
            features[t],
            feature_indices,
            k=k_homeostasis,
        )

        if not np.isfinite(h_prev) or not np.isfinite(h_cur):
            continue

        # actions[t] is the action a_{t-1} that produced transition s_{t-1} -> s_t.
        a_mag = action_magnitude(int(actions[t]), n_bins=n_bins)
        e_t = h_cur - h_prev - alpha * a_mag
        efficiencies.append(e_t)

    E_efficiency = float(np.mean(efficiencies)) if efficiencies else np.nan

    return cumulative_reward, G_survival, U_uncertainty, E_efficiency


def evaluate_candidate_fitness(
    episodes: Dict[str, Dict[str, np.ndarray]],
    feature_indices: Dict[str, int],
    candidate: Dict[str, Any],
    n_bins: int = 5,
    gamma: float = 0.99,
    sofa_eps: float = 2.0,
    alpha: float = 0.1,
    k_homeostasis: float = 10.0,
) -> Dict[str, float]:
    R_vals = []
    G_vals = []
    U_vals = []
    E_vals = []

    iterator = tqdm(
        episodes.values(),
        desc=f"Evaluating {candidate.get('name', 'candidate')}",
        leave=False,
    )

    for ep in iterator:
        R, G, U, E = compute_episode_fitness_terms(
            ep=ep,
            feature_indices=feature_indices,
            candidate=candidate,
            n_bins=n_bins,
            gamma=float(candidate.get("gamma", gamma)),
            sofa_eps=float(candidate.get("jsurv_epsilon", sofa_eps)),
            alpha=float(candidate.get("jcomp_alpha", alpha)),
            k_homeostasis=float(candidate.get("jcomp_k", k_homeostasis)),
        )

        R_vals.append(R)
        G_vals.append(G)
        U_vals.append(U)
        E_vals.append(E)

    R_vals = np.asarray(R_vals, dtype=np.float64)
    G_vals = np.asarray(G_vals, dtype=np.float64)
    U_vals = np.asarray(U_vals, dtype=np.float64)
    E_vals = np.asarray(E_vals, dtype=np.float64)

    return {
        "Jsurv": safe_corr(R_vals, G_vals),
        "Jconf": -safe_corr(R_vals, U_vals),
        "Jcomp": safe_corr(R_vals, E_vals),
        "R_mean": float(np.nanmean(R_vals)),
        "R_std": float(np.nanstd(R_vals)),
        "G_mean": float(np.nanmean(G_vals)),
        "U_mean": float(np.nanmean(U_vals)),
        "E_mean": float(np.nanmean(E_vals)),
        "num_episodes": int(len(R_vals)),
    }


def pareto_front(df: pd.DataFrame, objective_cols: List[str]) -> pd.DataFrame:
    values = df[objective_cols].to_numpy(dtype=np.float64)

    valid = np.all(np.isfinite(values), axis=1)
    is_pareto = np.zeros(len(df), dtype=bool)

    valid_indices = np.where(valid)[0]
    valid_values = values[valid]

    for local_i, global_i in enumerate(valid_indices):
        v = valid_values[local_i]

        dominated = np.any(
            np.all(valid_values >= v, axis=1)
            & np.any(valid_values > v, axis=1)
        )

        is_pareto[global_i] = not dominated

    return df[is_pareto].copy()


def add_selection_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    objective_cols = ["Jsurv", "Jconf", "Jcomp"]

    for col in objective_cols:
        vals = out[col].astype(float)
        mn, mx = vals.min(), vals.max()

        if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
            out[f"{col}_norm"] = (vals - mn) / (mx - mn)
        else:
            out[f"{col}_norm"] = 0.0

    out["utopia_distance"] = np.sqrt(
        (1.0 - out["Jsurv_norm"]) ** 2
        + (1.0 - out["Jconf_norm"]) ** 2
        + (1.0 - out["Jcomp_norm"]) ** 2
    )

    out["mean_fitness"] = out[objective_cols].mean(axis=1)
    return out


def load_candidates(path: str | None) -> List[Dict[str, Any]]:
    if path is None:
        print("No --candidates-json provided. Using DEFAULT_CANDIDATES.")
        return DEFAULT_CANDIDATES

    with open(path, "r") as f:
        candidates = json.load(f)

    if not isinstance(candidates, list):
        raise ValueError("Candidates JSON must be a list of dictionaries.")

    for i, cand in enumerate(candidates):
        if not isinstance(cand, dict):
            raise ValueError(f"Candidate {i} is not a dictionary.")
        if "name" not in cand:
            cand["name"] = f"candidate_{i:03d}"

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--candidates-json", default=None)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--pareto-csv", default=None)

    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--sofa-eps", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--k-homeostasis", type=float, default=10.0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("Loading episodes...")
    episodes = load_episodes(args.episodes_dir)
    print(f"Loaded episodes: {len(episodes)}")

    episodes = subsample_episodes(
        episodes=episodes,
        max_episodes=args.max_episodes,
        seed=args.seed,
    )
    print(f"Using episodes: {len(episodes)}")

    print("\nLoading feature columns...")
    feature_cols = load_feature_cols(args.dataset_dir)

    feature_indices = find_feature_indices(feature_cols)

    if len(feature_indices) == 0:
        raise ValueError("No critical features found. Check feature aliases / column_config.pkl.")

    print("\nLoading candidates...")
    candidates = load_candidates(args.candidates_json)
    print(f"Loaded candidates: {len(candidates)}")

    rows = []

    for candidate in tqdm(candidates, desc="Candidates"):
        metrics = evaluate_candidate_fitness(
            episodes=episodes,
            feature_indices=feature_indices,
            candidate=candidate,
            n_bins=args.num_bins,
            gamma=args.gamma,
            sofa_eps=args.sofa_eps,
            alpha=args.alpha,
            k_homeostasis=args.k_homeostasis,
        )

        row = dict(candidate)
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = add_selection_scores(df)

    df_sorted = df.sort_values(
        ["utopia_distance", "mean_fitness"],
        ascending=[True, False],
    )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df_sorted.to_csv(args.output_csv, index=False)

    print("\n" + "=" * 80)
    print("BEST CANDIDATES")
    print("=" * 80)
    print(
        df_sorted[
            [
                "name",
                "Jsurv",
                "Jconf",
                "Jcomp",
                "mean_fitness",
                "utopia_distance",
                "R_mean",
                "R_std",
            ]
        ].head(10).to_string(index=False)
    )

    pf = pareto_front(df_sorted, ["Jsurv", "Jconf", "Jcomp"])
    pf = pf.sort_values("utopia_distance", ascending=True)

    if args.pareto_csv is not None:
        os.makedirs(os.path.dirname(args.pareto_csv) or ".", exist_ok=True)
        pf.to_csv(args.pareto_csv, index=False)
        print(f"\nSaved Pareto front: {args.pareto_csv}")

    print(f"\nSaved all candidate metrics: {args.output_csv}")

    if len(pf) > 0:
        print("\n" + "=" * 80)
        print("BEST PARETO CANDIDATE")
        print("=" * 80)
        print(
            pf[
                [
                    "name",
                    "Jsurv",
                    "Jconf",
                    "Jcomp",
                    "mean_fitness",
                    "utopia_distance",
                    "action_cost_scale",
                    "confidence_tau",
                    "half_life",
                ]
            ].head(1).to_string(index=False)
        )


if __name__ == "__main__":
    main()