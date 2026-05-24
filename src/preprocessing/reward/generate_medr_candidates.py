import argparse
import itertools
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    action_cost_scales = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    half_lives = [24.0, 48.0, 72.0, 96.0]
    confidence_taus = [6.0, 12.0, 24.0]
    sigma_scales = [0.75, 1.0, 1.25, 1.5]
    k_scales = [0.5, 0.75, 1.0, 1.25]
    survival_confidence_weights = [
            (0.6, 0.4),
            (0.7, 0.3),
            (0.8, 0.2),
        ]

    grid = list(itertools.product(
        action_cost_scales,
        half_lives,
        confidence_taus,
        sigma_scales,
        k_scales,
        survival_confidence_weights,
    ))

    random.shuffle(grid)
    grid = grid[:args.max_candidates]

    candidates = []

    for i, (acs, hl, tau, sig, k, weights) in enumerate(grid):
        survival_w, confidence_w = weights
        candidates.append({
            "name": f"medr_cand_{i:04d}",
            "action_cost_scale": acs,
            "half_life": hl,
            "confidence_tau": tau,
            "sigma_scale": sig,
            "k_scale": k,
            "survival_weight": survival_w,
            "confidence_weight": confidence_w,
            "gamma": 0.99,
            "jcomp_alpha": 0.1,
            "jcomp_k": 10.0,
            "jsurv_epsilon": 2.0,
        })

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w") as f:
        json.dump(candidates, f, indent=2)

    print(f"Saved {len(candidates)} candidates to: {args.output_json}")


if __name__ == "__main__":
    main()