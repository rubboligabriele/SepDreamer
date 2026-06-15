import argparse
import os
import pathlib
import pickle
import sys
import time

from ruamel.yaml import YAML
from datetime import datetime
from sklearn.model_selection import train_test_split

import src.meddreamer.utils.tools as tools
from src.meddreamer.dreamer import Dreamer, MedDreamer

_FEAT_NAMES = None  # loaded per-run from column_config.pkl (see main())

def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.train_batch_length, seed=config.seed)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset

def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    # Load feature_names from column_config.pkl to match the exact episode column order.
    # Using ALL_FEATURE_COLUMNS would give a different ordering and cause feature-name/data mismatches.
    if not hasattr(config, "feature_names"):
        col_cfg_path = os.path.join(config.datadir, config.dataset, "column_config.pkl")
        if os.path.exists(col_cfg_path):
            with open(col_cfg_path, "rb") as _f:
                _col_cfg = pickle.load(_f)
            config.feature_names = list(_col_cfg["feature_cols"][: config.num_features])
            print(f"[main] feature_names loaded from column_config.pkl ({len(config.feature_names)} features)", flush=True)
        else:
            print(f"[main] WARNING: column_config.pkl not found at {col_cfg_path}; feature names will be generic", flush=True)

    logdir = pathlib.Path(config.logdir).expanduser() / f"{datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}_{config.logname}_{config.dataset}_{config.task}"

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    logger = tools.Logger(logdir)
    
    start_time = time.time()

    eps_dir = os.path.join(config.datadir, config.dataset, 'episodes')

    all_stay_ids = tools.load_all_episode_keys(eps_dir)
    if config.debug:
        all_stay_ids = all_stay_ids[:1000]

    # where to save the splits for reproducibility
    cache_root = os.path.dirname(eps_dir)
    split_path = os.path.join(cache_root, f"splits_seed{config.seed}.pkl")

    # if piclkes with splits already exist, load them, otherwise create new splits and save them
    if os.path.exists(split_path):
        print(f"Loading fixed splits from {split_path}", flush=True)
        with open(split_path, "rb") as f:
            splits = pickle.load(f)
        train_stay_ids = splits["train"]
        val_stay_ids = splits["val"]
        test_stay_ids = splits["test"]
    else:
        trainval_stay_ids, test_stay_ids = train_test_split(
            all_stay_ids, train_size=0.8, random_state=config.seed
        )

        train_stay_ids, val_stay_ids = train_test_split(
            trainval_stay_ids, test_size=0.125, random_state=config.seed
        )

        splits = {
            "train": train_stay_ids,
            "val": val_stay_ids,
            "test": test_stay_ids,
        }

        with open(split_path, "wb") as f:
            pickle.dump(splits, f)

        print(f"Saved fixed splits to {split_path}", flush=True)

    cache_root = os.path.dirname(eps_dir)

    # chache files names
    train_cache_path = os.path.join(cache_root, f"train_eps_cache_seed{config.seed}.pkl")
    val_cache_path = os.path.join(cache_root, f"val_eps_cache_seed{config.seed}.pkl")
    test_cache_path = os.path.join(cache_root, f"test_eps_cache_seed{config.seed}.pkl")

    if config.training:
        train_eps = tools.load_split_episodes(eps_dir, train_stay_ids, cache_path=train_cache_path)
        eval_eps = tools.load_split_episodes(eps_dir, val_stay_ids, cache_path=val_cache_path)

        # create a geenrator that samples from the training episodes with the specified batch length,
        # and then create a tf dataset from that generator with the specified batch size
        train_dataset = make_dataset(train_eps, config)
        print(f"Using full validation set for eval: {len(eval_eps)} episodes.", flush=True)
    else:
        train_eps = None
        train_dataset = None
        eval_eps = tools.load_split_episodes(eps_dir, test_stay_ids, cache_path=test_cache_path)
        print("Using test split for final evaluation.", flush=True)

    print(f"Train stays: {len(train_stay_ids)}", flush=True)
    print(f"Val stays: {len(val_stay_ids)}", flush=True)
    print(f"Test stays: {len(test_stay_ids)}", flush=True)
    
    end_time = time.time()
    print(f"Time taken to load data: {(end_time - start_time)/60:.2f} min.")

    if config.mode == "dreamer":
        agent = Dreamer(config, logger, logdir, train_dataset, eval_eps).to(config.device)
        if config.training:
            agent.train(config.epochs)
        else:
            for epoch in range(2000, config.epochs + 1, config.save_every):
                tools.load_model(agent, "all", config.ckptdir, epoch, config.device)
                agent.eval(eval_eps, epoch)

    else:
        agent = MedDreamer(config, logger, logdir, train_dataset, eval_eps).to(config.device)

        if config.mode == "world_model":
            if config.training:
                agent.train_wm(config.epochs)
            else:
                if config.ckptepoch > 0:
                    tools.load_model(agent, "wm", config.ckptdir, config.ckptepoch, config.device)
                    agent.eval_wm(eval_eps, config.ckptepoch)
                else:
                    for epoch in range(config.save_every, config.epochs + 1, config.save_every):
                        tools.load_model(agent, "wm", config.ckptdir, epoch, config.device)
                        agent.eval_wm(eval_eps, epoch)

        elif config.mode == "behavior":
            if config.training:
                tools.load_model(agent, "wm", config.ckptdir, config.ckptepoch, config.device)
                agent.train_behavior(config.epochs)
            else:
                for epoch in range(config.save_every, config.epochs + 1, config.save_every):
                    tools.load_model(agent, "behavior_policy", config.ckptdir, epoch, config.device)
                    agent._eval_behavior(eval_eps)

        elif config.mode == "policy_p1":
            if config.training:
                tools.load_model(agent, "wm", config.ckptdir, config.ckptepoch, config.device)
                tools.load_model(agent, "behavior_policy", config.behavior_ckptdir, config.behavior_ckptepoch, config.device)
                agent.train_policy(config.epochs, use_history=True)
            else:
                tools.load_model(agent, "all", config.ckptdir, config.ckptepoch, config.device)
                agent.eval(eval_eps, config.ckptepoch)

        elif config.mode == "policy_p2":
            if config.training:
                tools.load_model(agent, "all", config.ckptdir, config.ckptepoch, config.device, config.actor["lr"], config.critic["lr"])
                agent.train_policy(config.epochs, use_history=False)
            else:
                tools.load_model(agent, "all", config.ckptdir, config.ckptepoch, config.device)
                agent.eval(eval_eps, config.ckptepoch)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    yaml = YAML(typ='rt')
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))