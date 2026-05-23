import argparse
import itertools
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    action_cost_scales = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02]
    half_lives = [24.0, 48.0, 72.0, 96.0]
    confidence_taus = [6.0, 12.0, 24.0]
    sigma_scales = [0.75, 1.0, 1.25, 1.5]
    k_scales = [0.5, 0.75, 1.0, 1.25]

    candidates = []

    for i, (acs, hl, tau, sig, k) in enumerate(
        itertools.product(
            action_cost_scales,
            half_lives,
            confidence_taus,
            sigma_scales,
            k_scales,
        )
    ):
        candidates.append(
            {
                "name": f"medr_cand_{i:04d}",
                "action_cost_scale": acs,
                "half_life": hl,
                "confidence_tau": tau,
                "sigma_scale": sig,
                "k_scale": k,
                "gamma": 0.99,
                "jcomp_alpha": 0.1,
                "jcomp_k": 10.0,
                "jsurv_epsilon": 2.0,
            }
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(candidates, f, indent=2)

    print(f"Saved {len(candidates)} candidates to: {args.output}")


if __name__ == "__main__":
    main()