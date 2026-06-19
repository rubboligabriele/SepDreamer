import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import trange
from tqdm import tqdm

import src.meddreamer.models.models as models
import src.meddreamer.utils.tools as tools

import torch
from torch import nn


class Dreamer(nn.Module):
    def __init__(self, config, logger, logdir, train_dataset, eval_dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._logdir = logdir
        self._should_log = tools.Every(config.log_every)
        self._should_eval = tools.Every(config.eval_every)
        self._should_save = tools.Every(config.save_every)
        self._metrics = {}
        self._images = {}
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._wm = models.WorldModel(config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        self._behavior_policy = models.BehaviorPolicy(config)
        self._behavior_policy_loaded = False

    @staticmethod
    def _expand_episode(data):
        return {k: np.expand_dims(v, axis=0) for k, v in data.items()}

    @staticmethod
    def _augment_traj_debug(traj_ope, clin_action_list, stay_id):
        max_steps = min(len(clin_action_list), len(traj_ope["rho"]))
        actions_used = np.array(clin_action_list[:max_steps], dtype=np.int64)
        traj_ope["debug"]["stay_id"] = str(stay_id)
        traj_ope["debug"]["actions"] = actions_used.tolist()
        traj_ope["debug"]["frac_action_0"] = float(np.mean(actions_used == 0))
        traj_ope["debug"]["num_action_0"] = int(np.sum(actions_used == 0))
        traj_ope["debug"]["num_steps"] = int(len(actions_used))
        return traj_ope

    @staticmethod
    def _cont_penalty(cont_probs, a=0.01, b=0.01, c=0.001):
        weights = torch.tensor([-a, b, -c], device=cont_probs.device, dtype=cont_probs.dtype)
        return (cont_probs * weights).sum(dim=-1)

    def _set_eval_mode(self):
        self._wm.eval()
        self._task_behavior.eval()
        if self._behavior_policy_loaded:
            self._behavior_policy.eval()

    def _run_ai_rollout(self, full_states, T_roll):
        state_ai = {k: v[:, 4] for k, v in full_states.items()}
        ai_episode_return = 0.0
        actions = []
        ai_probs = []

        for t in range(T_roll):
            feat_ai = self._wm.dynamics.get_feat(state_ai)
            ai_dist = self._task_behavior.actor(feat_ai.detach())
            ai_probs.append(tools.to_np(ai_dist.probs.squeeze(0)))
            action = ai_dist.sample()
            actions.append(tools.to_np(action))
            state_ai = self._wm.dynamics.img_step(state_ai, action)
            reward_ai = self._wm.heads["reward"](
                self._wm.dynamics.get_feat(state_ai).detach()
            ).mode()
            ai_episode_return += float(tools.to_np(reward_ai.squeeze()))

        actions = np.stack(actions, axis=1)
        ai_actions_ep = np.argmax(np.squeeze(actions, axis=0), axis=-1)
        ai_probs_np = np.stack(ai_probs, axis=0)

        return ai_episode_return, ai_actions_ep, ai_probs_np

    def _compute_ope_loop(self, full_states, phys_action, pi_b_seq, data):
        pi_ai_clin_list = []
        pi_b_clin_list = []
        reward_list = []
        clin_action_list = []

        for t in range(5, phys_action.shape[1] - 1):
            real_state_t = {k: v[:, t] for k, v in full_states.items()}
            feat_real = self._wm.dynamics.get_feat(real_state_t)
            clin_action_onehot = phys_action[:, t + 1]
            clin_action_list.append(int(torch.argmax(clin_action_onehot[0]).item()))

            ai_dist_real = self._task_behavior.actor(feat_real.detach())
            pi_ai = torch.exp(ai_dist_real.log_prob(clin_action_onehot))
            pi_b = pi_b_seq[:, t]

            pi_ai_clin_list.append(float(tools.to_np(pi_ai.squeeze())))
            pi_b_clin_list.append(float(tools.to_np(pi_b.squeeze())))
            reward_list.append(float(tools.to_np(data["reward"][:, t + 1].squeeze())))

        return pi_ai_clin_list, pi_b_clin_list, reward_list, clin_action_list

    def _compute_pi_b_seq(self, full_states, phys_action, data):
        if self._behavior_policy.input_type == "raw":
            feat_seq = data["features"][:, :-1]
        else:
            feat_seq = self._wm.dynamics.get_feat(full_states)[:, :-1]
        act_seq = phys_action[:, 1:]
        dist_b_seq = self._behavior_policy(feat_seq, data["is_first"][:, :-1])
        return torch.exp(dist_b_seq.log_prob(act_seq))

    def _update_metrics(self, metrics):
        for name, value in metrics.items():
            self._metrics.setdefault(name, []).append(value)

    def _finalize_ope(self, ope_trajs, ai_episode_returns, phys_episode_returns, mortalities, value_estimates, true_mortality, valid_episodes, imag_rewards, ai_sample_counts):
        phys_episode_returns = np.array(phys_episode_returns, dtype=np.float32)
        ai_episode_returns = np.array(ai_episode_returns, dtype=np.float32)
        value_estimates = np.array(value_estimates, dtype=np.float32)
        mortalities = np.array(mortalities, dtype=np.float32)

        print("\n[GLOBAL AI SAMPLE ACTION COUNTS]", flush=True)
        print(ai_sample_counts.tolist(), flush=True)
        print("sample_argmax:", int(ai_sample_counts.argmax()), flush=True)
        print("sample_argmax_frac:", float(ai_sample_counts.max() / max(ai_sample_counts.sum(), 1)), flush=True)

        fig, bin_centers, smoothed, smoothed_sem = tools.plot_mortality_vs_value(
            phys_episode_returns, mortalities, xlabel="Expected Return"
        )
        fig_value, _, _, _ = tools.plot_mortality_vs_value(
            value_estimates, mortalities, xlabel="Critic Value"
        )

        ope_summary = {
            "imag_episode_return": float(imag_rewards) / valid_episodes,
            "ai_episode_return": float(ai_episode_returns.mean()),
            "critic_value_mean": float(value_estimates.mean()),
            "critic_value_std": float(value_estimates.std()),
        }

        if ope_trajs:
            tools.debug_ope_summary(ope_trajs)
            ope_metrics = tools.finalize_ope(ope_trajs, debug=True, top_k=10)
            ai_mortality, _ = tools.calculate_estimated_mortality(
                ai_episode_returns, bin_centers, smoothed, smoothed_sem
            )
            true_mortality = true_mortality / valid_episodes
            mortality_decrease = true_mortality - ai_mortality
            ope_summary.update({
                "ai_mortality": round(ai_mortality * 100, 2),
                "true_mortality": round(true_mortality * 100, 2),
                "mortality_decrease": round(mortality_decrease * 100, 2),
                "wis": ope_metrics["wis"],
                "wpdis": ope_metrics["wpdis"],
                "cwpdis": ope_metrics["cwpdis"],
                "ess": ope_metrics["ess"],
            })

        return ope_summary, fig, fig_value, phys_episode_returns, ai_episode_returns, mortalities, value_estimates

    def _get_policy_value_estimate(self, feat):
        return self._task_behavior.value(feat).mode().squeeze(-1)

    def train(self, epochs):
        for epoch in trange(0, epochs + 1, desc="Training"):
            post, _, data = self._wm._load(next(self._train_dataset))
            feat = self._wm.dynamics.get_feat(post)
            self._train_policy(post, feat, data, use_history=True)
            self._eval_log("all", epoch)

    def eval(self, episodes, epoch):
        metrics = {"step": epoch}
        imag_rewards = 0
        true_mortality = 0
        phys_episode_returns = []
        ai_episode_returns = []
        value_estimates = []
        ai_actions = []
        phys_actions = []
        sofas = []
        mortalities = []
        full_mort = []
        valid_episodes = 0
        ope_trajs = []
        ai_sample_counts = np.zeros(self._config.num_actions, dtype=np.int64)

        self._set_eval_mode()
        first_debug_done = False

        with torch.no_grad():
            for stay_id, data in tqdm(episodes.items()):
                data = self._expand_episode(data)
                B, T, _ = data["features"].shape
                data = self._wm.preprocess(data)

                phys_action = data["action"].detach().clone()
                is_first = data["is_first"].detach().clone()

                if phys_action.shape[1] <= 5:
                    print(f"Skipping short episode {stay_id} with length {phys_action.shape[1]}", flush=True)
                    continue

                valid_episodes += 1
                debug_now = not first_debug_done

                if debug_now:
                    print(f"\n{'='*80}")
                    print(f"[EVAL DEBUG] FIRST STAY = {stay_id}")
                    print(f"{'='*80}\n[FIRST STAY ALIGNMENT TABLE]")
                    for t in range(min(12, phys_action.shape[1])):
                        print(
                            f"t={t:02d} action_idx={int(torch.argmax(phys_action[0, t]).item()):02d} "
                            f"reward={float(data['reward'][0, t].item()):8.4f} "
                            f"mortality={float(data['mortality'][0, t].item()):.0f}"
                        )

                features = tools.bt_flatten(data["features"])
                delta = tools.bt_flatten(data["delta"]) if self._config.fm["use_fm"] else None
                embed = tools.bt_unflatten(self._wm.encoder(features, delta), B, T)

                self._wm.dynamics._debug_mode = debug_now
                if debug_now:
                    self._wm.dynamics._debug_obs_counter = 0
                full_states, _ = self._wm.dynamics.observe(embed, phys_action, is_first, debug=debug_now)
                self._wm.dynamics._debug_mode = False
                pi_b_seq = self._compute_pi_b_seq(full_states, phys_action, data) if self._behavior_policy_loaded else None

                init = {k: v[:, 4] for k, v in full_states.items()}
                feat_init = self._wm.dynamics.get_feat(init)
                value_estimates.append(float(tools.to_np(self._get_policy_value_estimate(feat_init.detach()).squeeze())))

                if debug_now:
                    print("\n[ROLLOUT START] init state from t=4")
                    for i in range(min(8, phys_action[:, 5:].shape[1])):
                        print(f"roll_step={i:02d} uses phys_action at t={5+i:02d} -> action_idx={int(torch.argmax(phys_action[0, 5+i]).item()):02d}")
                    self._wm.dynamics._debug_img_mode = True
                    self._wm.dynamics._debug_img_counter = 0
                else:
                    self._wm.dynamics._debug_img_mode = False

                prior = self._wm.dynamics.imagine_with_action(phys_action[:, 5:], init)
                self._wm.dynamics._debug_img_mode = False

                phys_episode_returns.append(tools.to_np(data["reward"][:, 5:].sum(dim=1).squeeze()))
                mortalities.append(tools.to_np(data["mortality"][:, 0].squeeze()))

                imag_feat, imag_state, imag_action = self._task_behavior._imagine_in_time(
                    init, self._task_behavior.actor, 50
                )
                imag_rewards += float(tools.to_np(
                    self._wm.heads["reward"](self._wm.dynamics.get_feat(imag_state)).mode().sum()
                ))

                T_roll = prior["stoch"].shape[1]

                ai_episode_return, ai_actions_ep, ai_probs_np = self._run_ai_rollout(full_states, T_roll)
                ai_actions.append(ai_actions_ep)
                ai_sample_counts += np.bincount(ai_actions_ep, minlength=self._config.num_actions)

                if debug_now:
                    print("\n[EVAL AI POLICY PROBS DEBUG]")
                    print("mean_probs:", np.round(ai_probs_np.mean(axis=0), 4).tolist())
                    print("mean_probs_max:", float(ai_probs_np.mean(axis=0).max()))
                    print("mean_probs_argmax:", int(ai_probs_np.mean(axis=0).argmax()))

                phys_actions.append(np.argmax(tools.to_np(phys_action[0, 5:]), axis=-1))
                if "sofa" in data:
                    sofas.append(tools.to_np(data["sofa"][0, 5:]))
                full_mort.append(tools.to_np(data["mortality"][0, 5:]))
                ai_episode_returns.append(ai_episode_return)

                if self._behavior_policy_loaded:
                    pi_ai_clin_list, pi_b_clin_list, reward_list, clin_action_list = self._compute_ope_loop(
                        full_states, phys_action, pi_b_seq, data
                    )
                    traj_ope = tools.compute_ope_trajectory(
                        pi_ai_clin_list, pi_b_clin_list, reward_list,
                        gamma=self._config.discount, prob_eps=1e-6, rho_max=5.0, max_ope_steps=30,
                    )
                    if traj_ope is not None:
                        ope_trajs.append(self._augment_traj_debug(traj_ope, clin_action_list, stay_id))

                if data["mortality"].any():
                    true_mortality += 1
                if debug_now:
                    first_debug_done = True

        if valid_episodes == 0:
            print("No valid episodes for evaluation.", flush=True)
            return

        ope_summary, fig, fig_value, phys_episode_returns, ai_episode_returns, mortalities, value_estimates = \
            self._finalize_ope(ope_trajs, ai_episode_returns, phys_episode_returns,
                               mortalities, value_estimates, true_mortality,
                               valid_episodes, imag_rewards, ai_sample_counts)
        if ope_summary:
            metrics.update(ope_summary)
        fig_value.savefig(os.path.join(self._logdir, f"mortality_vs_value_{epoch}.png"))

        rows_mort, rows_phys, rows_ai, rows_sofa = [], [], [], []
        for i, (mort_arr, phys_arr, ai_arr) in enumerate(zip(full_mort, phys_actions, ai_actions)):
            ep_mort = float(mort_arr.max()) if hasattr(mort_arr, "max") else float(mort_arr[0])
            T = min(len(phys_arr), len(ai_arr))
            for t in range(T):
                rows_mort.append(ep_mort)
                rows_phys.append(int(phys_arr[t]))
                rows_ai.append(int(ai_arr[t]))
                if len(sofas) > i:
                    rows_sofa.append(float(sofas[i][t]) if t < len(sofas[i]) else float("nan"))
        data_out = {"mortality": rows_mort, "phys_action": rows_phys, "ai_action": rows_ai}
        if rows_sofa:
            data_out["sofa"] = rows_sofa
        pd.DataFrame(data_out).to_csv(os.path.join(self._logdir, f"result_data_{epoch}.csv"), index=False)

        np.savez(
            os.path.join(self._logdir, f"phys_and_mortality_{epoch}.npz"),
            phys=phys_episode_returns, mort=mortalities, value=value_estimates,
        )
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")

    def eval_wm(self, episodes, epoch):
        phys_episode_returns = []
        mortalities = []
        features_dict = {"ori_feat": [], "recon_feat": []}
        first_debug_done = False
        valid_episodes = 0

        recon_error = 0.0
        reward_nll_post = 0.0
        reward_nll_prior = 0.0

        # per-feature reconstruction accumulators
        num_features = getattr(self._config, "num_features", None)
        _nf = num_features if num_features is not None else 0
        feat_mse_post_sum   = np.zeros(_nf, dtype=np.float64)
        feat_mse_prior_sum  = np.zeros(_nf, dtype=np.float64)
        feat_mask_post_cnt  = np.zeros(_nf, dtype=np.float64)
        feat_mask_prior_cnt = np.zeros(_nf, dtype=np.float64)
        feat_mse_post_death_sum   = np.zeros(_nf, dtype=np.float64)
        feat_mse_post_surv_sum    = np.zeros(_nf, dtype=np.float64)
        feat_mask_post_death_cnt  = np.zeros(_nf, dtype=np.float64)
        feat_mask_post_surv_cnt   = np.zeros(_nf, dtype=np.float64)
        feat_mse_prior_death_sum  = np.zeros(_nf, dtype=np.float64)
        feat_mse_prior_surv_sum   = np.zeros(_nf, dtype=np.float64)
        feat_mask_prior_death_cnt = np.zeros(_nf, dtype=np.float64)
        feat_mask_prior_surv_cnt  = np.zeros(_nf, dtype=np.float64)

        reward_mae_post_sum = 0.0
        reward_mae_post_count = 0
        reward_mae_prior_sum = 0.0
        reward_mae_prior_count = 0
        reward_action_range_list = []
        reward_action_std_list = []
        reward_action_best_list = []
        cont_action_best_list = []
        reward_action0_rank_list = []

        cont_correct = 0
        cont_total = 0
        cont_pos_correct = 0
        cont_pos_total = 0
        cont_neg_correct = 0
        cont_neg_total = 0
        cont_prob_on_pos = []
        cont_prob_on_neg = []

        cont_class_correct = {0: 0, 1: 0, 2: 0}
        cont_class_total = {0: 0, 1: 0, 2: 0}
        cont_pred_class_probs = {0: [], 1: [], 2: []}

        mort3_terminal_correct = 0
        mort3_terminal_total = 0
        mort3_nonterminal_correct = 0
        mort3_nonterminal_total = 0
        mort3_death_terminal_correct = 0
        mort3_death_terminal_total = 0
        mort3_survival_terminal_correct = 0
        mort3_survival_terminal_total = 0

        self._wm.eval()

        with torch.no_grad():
            for stay_id, data in tqdm(episodes.items()):
                data = self._expand_episode(data)
                B, T, _ = data["features"].shape
                data = self._wm.preprocess(data)

                phys_action = data["action"].detach().clone()
                is_first = data["is_first"].detach().clone()

                debug_now = not first_debug_done

                if debug_now:
                    print(f"\n{'=' * 80}")
                    print(f"[EVAL_WM DEBUG] FIRST STAY = {stay_id}")
                    print(f"{'=' * 80}\n[FIRST STAY ALIGNMENT TABLE]")
                    for t in range(min(12, phys_action.shape[1])):
                        print(
                            f"t={t:02d} action_idx={int(torch.argmax(phys_action[0, t]).item()):02d} "
                            f"reward={float(data['reward'][0, t].item()):8.4f} "
                            f"mortality={float(data['mortality'][0, t].item()):.0f} "
                            f"is_terminal={float(data['is_terminal'][0, t].item()):.0f}",
                            flush=True,
                        )

                if phys_action.shape[1] <= 5:
                    print(f"Skipping short episode {stay_id} with length {phys_action.shape[1]}", flush=True)
                    continue

                valid_episodes += 1

                features = tools.bt_flatten(data["features"])
                delta = tools.bt_flatten(data["delta"]) if self._config.fm["use_fm"] else None
                embed = tools.bt_unflatten(self._wm.encoder(features, delta), B, T)

                self._wm.dynamics._debug_mode = debug_now
                if debug_now:
                    self._wm.dynamics._debug_obs_counter = 0
                states, _ = self._wm.dynamics.observe(
                    embed[:, :5], phys_action[:, :5], is_first[:, :5], debug=debug_now,
                )
                self._wm.dynamics._debug_mode = False

                init = {k: v[:, -1] for k, v in states.items()}

                if debug_now:
                    print("\n[ROLLOUT START FROM t=4]")
                    for i in range(min(8, phys_action[:, 5:].shape[1])):
                        print(f"roll_step={i:02d} uses phys_action at t={5+i:02d} -> action_idx={int(torch.argmax(phys_action[0, 5+i]).item()):02d}", flush=True)
                    self._wm.dynamics._debug_img_mode = True
                    self._wm.dynamics._debug_img_counter = 0
                else:
                    self._wm.dynamics._debug_img_mode = False

                prior = self._wm.dynamics.imagine_with_action(phys_action[:, 5:], init)
                self._wm.dynamics._debug_img_mode = False

                feat_post = self._wm.dynamics.get_feat(states)
                feat_prior = self._wm.dynamics.get_feat(prior)

                # WM ACTION SENSITIVITY: one-step from real state t=4
                state_probe = {k: v[:, -1] for k, v in states.items()}
                rewards_by_action = []
                cont_by_action = []

                for a in range(self._config.num_actions):
                    action_a = torch.zeros((1, self._config.num_actions), device=self._config.device, dtype=torch.float32)
                    action_a[:, a] = 1.0
                    next_state_a = self._wm.dynamics.img_step(state_probe, action_a, sample=False)
                    feat_a = self._wm.dynamics.get_feat(next_state_a)
                    rewards_by_action.append(float(self._wm.heads["reward"](feat_a).mode().squeeze().item()))
                    cont_by_action.append(float(self._wm.heads["cont"](feat_a).mode().squeeze().item()))

                rewards_by_action = np.array(rewards_by_action, dtype=np.float32)
                cont_by_action = np.array(cont_by_action, dtype=np.float32)

                reward_action_range_list.append(float(rewards_by_action.max() - rewards_by_action.min()))
                reward_action_std_list.append(float(rewards_by_action.std()))
                reward_action_best_list.append(int(rewards_by_action.argmax()))
                cont_action_best_list.append(int(cont_by_action.argmax()))

                rank_desc = np.argsort(-rewards_by_action)
                action0_rank = int(np.where(rank_desc == 0)[0][0]) + 1
                reward_action0_rank_list.append(action0_rank)

                if debug_now:
                    print("\n[WM ONE-STEP ACTION DEBUG]")
                    print("reward_by_action:", np.round(rewards_by_action, 4).tolist())
                    print("cont_by_action:", np.round(cont_by_action, 4).tolist())
                    print("best_reward_action:", int(rewards_by_action.argmax()))
                    print("best_cont_action:", int(cont_by_action.argmax()))
                    print("action0_reward_rank:", int(action0_rank))
                    print("reward_range:", float(rewards_by_action.max() - rewards_by_action.min()))
                    print("reward_std:", float(rewards_by_action.std()))

                    print("\n[WM MULTI-STEP SAME-ACTION DEBUG]")
                    for a in range(self._config.num_actions):
                        s = {k: v.clone() for k, v in state_probe.items()}
                        action_a = torch.zeros((1, self._config.num_actions), device=self._config.device, dtype=torch.float32)
                        action_a[:, a] = 1.0
                        rewards_roll, conts_roll = [], []
                        for h in range(5):
                            s = self._wm.dynamics.img_step(s, action_a, sample=False)
                            feat_h = self._wm.dynamics.get_feat(s)
                            rewards_roll.append(float(self._wm.heads["reward"](feat_h).mode().squeeze().item()))
                            conts_roll.append(float(self._wm.heads["cont"](feat_h).mode().squeeze().item()))
                        print(
                            f"a={a:02d} sum_r_5={sum(rewards_roll):8.4f} "
                            f"mean_cont_5={np.mean(conts_roll):.4f} "
                            f"r_seq={np.round(rewards_roll, 3).tolist()}",
                            flush=True,
                        )

                # Heads
                reward_head_post = self._wm.heads["reward"](feat_post)
                reward_head_prior = self._wm.heads["reward"](feat_prior)
                reward_post = reward_head_post.mode()
                reward_prior = reward_head_prior.mode()
                cont_head_post = self._wm.heads["cont"](feat_post)
                cont_head_prior = self._wm.heads["cont"](feat_prior)
                recon = self._wm.heads["decoder"](feat_post)["features"].mode()
                openl = self._wm.heads["decoder"](feat_prior)["features"].mode()

                # Reconstruction (bloc excluded — same as training)
                eval_mask = data["mask"].clone()
                eval_mask[..., -1] = 0
                model = torch.cat([recon[:, :5], openl], 1)
                error = ((model - data["features"]) ** 2) * eval_mask
                recon_error += (error.sum(dim=-1) / (eval_mask.sum(dim=-1) + 1e-8)).mean().item()

                # Per-feature MSE accumulation
                if _nf > 0:
                    mask_post  = eval_mask[:, :5]
                    mask_prior = eval_mask[:, 5:]
                    err_post  = ((recon[:, :5] - data["features"][:, :5]) ** 2) * mask_post
                    err_prior = ((openl       - data["features"][:, 5:]) ** 2) * mask_prior

                    feat_mse_post_sum  += tools.to_np(err_post.sum(dim=(0, 1)))
                    feat_mask_post_cnt += tools.to_np(mask_post.sum(dim=(0, 1)))
                    feat_mse_prior_sum  += tools.to_np(err_prior.sum(dim=(0, 1)))
                    feat_mask_prior_cnt += tools.to_np(mask_prior.sum(dim=(0, 1)))

                    is_death    = bool(data["mortality"][:, 0].any().item())
                    is_survival = not is_death
                    if is_death:
                        feat_mse_post_death_sum   += tools.to_np(err_post.sum(dim=(0, 1)))
                        feat_mask_post_death_cnt  += tools.to_np(mask_post.sum(dim=(0, 1)))
                        feat_mse_prior_death_sum  += tools.to_np(err_prior.sum(dim=(0, 1)))
                        feat_mask_prior_death_cnt += tools.to_np(mask_prior.sum(dim=(0, 1)))
                    else:
                        feat_mse_post_surv_sum   += tools.to_np(err_post.sum(dim=(0, 1)))
                        feat_mask_post_surv_cnt  += tools.to_np(mask_post.sum(dim=(0, 1)))
                        feat_mse_prior_surv_sum  += tools.to_np(err_prior.sum(dim=(0, 1)))
                        feat_mask_prior_surv_cnt += tools.to_np(mask_prior.sum(dim=(0, 1)))

                # Reward NLL
                reward_nll_post += (-reward_head_post.log_prob(data["reward"][:, :5])).mean().item()
                reward_nll_prior += (-reward_head_prior.log_prob(data["reward"][:, 5:])).mean().item()

                # Reward MAE
                true_reward_post = data["reward"][:, :5]
                true_reward_prior = data["reward"][:, 5:]
                mae_post = torch.abs(reward_post - true_reward_post)
                mae_prior = torch.abs(reward_prior - true_reward_prior)
                reward_mae_post_sum += mae_post.sum().item()
                reward_mae_post_count += mae_post.numel()
                reward_mae_prior_sum += mae_prior.sum().item()
                reward_mae_prior_count += mae_prior.numel()

                if debug_now:
                    print("\n[REWARD HEAD ALIGNMENT]")
                    for k in range(min(10, reward_prior.shape[1])):
                        real_t = 5 + k
                        print(
                            f"roll_step={k:02d} uses action at t={real_t:02d} "
                            f"act={int(torch.argmax(phys_action[0, real_t]).item()):02d} "
                            f"pred_reward={float(reward_prior[0, k].item()):8.4f} "
                            f"real_reward={float(data['reward'][0, real_t].item()):8.4f} "
                            f"is_terminal={int(data['is_terminal'][0, real_t].item())}",
                            flush=True,
                        )

                # Cont head analysis
                if self._config.cont_type in ["cont", "mort2"]:
                    cont_prob_all = torch.cat([cont_head_post.mean[:, :5], cont_head_prior.mean], dim=1)
                    cont_true_all = data["cont"].float()
                    cont_pred_all = (cont_prob_all >= 0.5).float()

                    cont_correct += (cont_pred_all == cont_true_all).sum().item()
                    cont_total += cont_true_all.numel()

                    pos_mask = cont_true_all == 1
                    neg_mask = cont_true_all == 0
                    if pos_mask.any():
                        cont_pos_correct += (cont_pred_all[pos_mask] == cont_true_all[pos_mask]).sum().item()
                        cont_pos_total += pos_mask.sum().item()
                        cont_prob_on_pos.extend(cont_prob_all[pos_mask].detach().cpu().view(-1).tolist())
                    if neg_mask.any():
                        cont_neg_correct += (cont_pred_all[neg_mask] == cont_true_all[neg_mask]).sum().item()
                        cont_neg_total += neg_mask.sum().item()
                        cont_prob_on_neg.extend(cont_prob_all[neg_mask].detach().cpu().view(-1).tolist())

                elif self._config.cont_type == "mort3":
                    cont_probs_all = torch.cat([cont_head_post.probs[:, :5], cont_head_prior.probs], dim=1)
                    cont_true_all = data["cont"]
                    true_cls = torch.argmax(cont_true_all, dim=-1)
                    pred_cls = torch.argmax(cont_probs_all, dim=-1)

                    cont_correct += (pred_cls == true_cls).sum().item()
                    cont_total += true_cls.numel()

                    for cls in [0, 1, 2]:
                        cls_mask = true_cls == cls
                        if cls_mask.any():
                            cont_class_correct[cls] += (pred_cls[cls_mask] == true_cls[cls_mask]).sum().item()
                            cont_class_total[cls] += cls_mask.sum().item()
                            cont_pred_class_probs[cls].extend(
                                cont_probs_all[..., cls][cls_mask].detach().cpu().view(-1).tolist()
                            )

                    terminal_mask = data["is_terminal"].bool()
                    nonterminal_mask = ~terminal_mask

                    if terminal_mask.any():
                        mort3_terminal_correct += (pred_cls[terminal_mask] == true_cls[terminal_mask]).sum().item()
                        mort3_terminal_total += terminal_mask.sum().item()
                    if nonterminal_mask.any():
                        mort3_nonterminal_correct += (pred_cls[nonterminal_mask] == true_cls[nonterminal_mask]).sum().item()
                        mort3_nonterminal_total += nonterminal_mask.sum().item()

                    death_terminal_mask = terminal_mask & (true_cls == 0)
                    survival_terminal_mask = terminal_mask & (true_cls == 1)
                    if death_terminal_mask.any():
                        mort3_death_terminal_correct += (pred_cls[death_terminal_mask] == true_cls[death_terminal_mask]).sum().item()
                        mort3_death_terminal_total += death_terminal_mask.sum().item()
                    if survival_terminal_mask.any():
                        mort3_survival_terminal_correct += (pred_cls[survival_terminal_mask] == true_cls[survival_terminal_mask]).sum().item()
                        mort3_survival_terminal_total += survival_terminal_mask.sum().item()

                phys_episode_returns.append(tools.to_np(reward_prior.sum(dim=1).squeeze()))
                mortalities.append(tools.to_np(data["mortality"][:, 0].squeeze()))

                recon_feature = torch.cat([recon[:, :5], openl], 1)
                features_dict["ori_feat"].append(tools.to_np(data["features"]))
                features_dict["recon_feat"].append(tools.to_np(recon_feature))

                if debug_now:
                    first_debug_done = True

        if valid_episodes == 0:
            print("No valid episodes for evaluation.", flush=True)
            return

        phys_episode_returns = np.array(phys_episode_returns, dtype=np.float32)
        mortalities = np.array(mortalities, dtype=np.float32)

        fig, bin_centers, smoothed, smoothed_sem = tools.plot_mortality_vs_value(
            phys_episode_returns, mortalities, xlabel="Expected Return (WM)"
        )
        fig.savefig(os.path.join(self._logdir, f"mortality_vs_expected_return_{epoch}.png"))

        # Per-feature reconstruction analysis
        if _nf > 0:
            feat_names = getattr(self._config, "feature_names", [f"feat_{i}" for i in range(_nf)])

            feat_mse_post  = feat_mse_post_sum  / np.maximum(feat_mask_post_cnt,  1)
            feat_mse_prior = feat_mse_prior_sum / np.maximum(feat_mask_prior_cnt, 1)

            if "urine_output" in feat_names:
                _uo_idx = feat_names.index("urine_output")
                print(f"\n[UO DEBUG epoch={epoch}]", flush=True)
                print(f"  urine_output MSE post={feat_mse_post[_uo_idx]:.4f}  prior={feat_mse_prior[_uo_idx]:.4f}", flush=True)
                # Sample raw true vs predicted values from last batch
                with torch.no_grad():
                    _uo_true_post  = data["features"][:, :5, _uo_idx]
                    _uo_pred_post  = recon[:, :5, _uo_idx]
                    _uo_true_prior = data["features"][:, 5:, _uo_idx]
                    _uo_pred_prior = openl[:, :, _uo_idx]
                    _mask_post     = data["mask"][:, :5, _uo_idx].bool()
                    _mask_prior    = data["mask"][:, 5:, _uo_idx].bool()
                    if _mask_post.any():
                        print(f"  POST  true:  {_uo_true_post[_mask_post][:8].cpu().tolist()}", flush=True)
                        print(f"  POST  pred:  {_uo_pred_post[_mask_post][:8].cpu().tolist()}", flush=True)
                    if _mask_prior.any():
                        print(f"  PRIOR true:  {_uo_true_prior[_mask_prior][:8].cpu().tolist()}", flush=True)
                        print(f"  PRIOR pred:  {_uo_pred_prior[_mask_prior][:8].cpu().tolist()}", flush=True)

            feat_mse_post_death  = feat_mse_post_death_sum  / np.maximum(feat_mask_post_death_cnt,  1)
            feat_mse_post_surv   = feat_mse_post_surv_sum   / np.maximum(feat_mask_post_surv_cnt,   1)
            feat_mse_prior_death = feat_mse_prior_death_sum / np.maximum(feat_mask_prior_death_cnt, 1)
            feat_mse_prior_surv  = feat_mse_prior_surv_sum  / np.maximum(feat_mask_prior_surv_cnt,  1)

            fig_post, fig_prior, fig_delta = tools.plot_recon_error_per_feature(
                feat_mse_post, feat_mse_prior, feat_names,
                feat_mse_post_death=feat_mse_post_death,
                feat_mse_prior_death=feat_mse_prior_death,
                feat_mse_post_surv=feat_mse_post_surv,
                feat_mse_prior_surv=feat_mse_prior_surv,
            )
            fig_post.savefig(os.path.join(self._logdir,  f"recon_mse_per_feat_post_{epoch}.png"))
            fig_prior.savefig(os.path.join(self._logdir, f"recon_mse_per_feat_prior_{epoch}.png"))
            fig_delta.savefig(os.path.join(self._logdir, f"recon_mse_per_feat_delta_{epoch}.png"))

            recon_feat_dict = {
                "feature_names": feat_names,
                "mse_post":  feat_mse_post.tolist(),
                "mse_prior": feat_mse_prior.tolist(),
                "mse_post_death":  feat_mse_post_death.tolist(),
                "mse_post_surv":   feat_mse_post_surv.tolist(),
                "mse_prior_death": feat_mse_prior_death.tolist(),
                "mse_prior_surv":  feat_mse_prior_surv.tolist(),
                "delta_prior_minus_post": (feat_mse_prior - feat_mse_post).tolist(),
                "worst_post_features":  [feat_names[i] for i in np.argsort(-feat_mse_post)[:10]],
                "worst_prior_features": [feat_names[i] for i in np.argsort(-feat_mse_prior)[:10]],
                "worst_delta_features": [feat_names[i] for i in np.argsort(-(feat_mse_prior - feat_mse_post))[:10]],
            }
            with open(os.path.join(self._logdir, f"recon_feat_mse_{epoch}.json"), "w") as f:
                json.dump(recon_feat_dict, f, indent=2)

            print("\n[RECON MSE PER FEATURE - TOP 10 WORST (post)]")
            for name in recon_feat_dict["worst_post_features"]:
                i = feat_names.index(name)
                print(f"  {name}: post={feat_mse_post[i]:.4f}  prior={feat_mse_prior[i]:.4f}  death={feat_mse_post_death[i]:.4f}  surv={feat_mse_post_surv[i]:.4f}")

            print("\n[RECON ROLLOUT DEGRADATION - TOP 10 (prior - post)]")
            for name in recon_feat_dict["worst_delta_features"]:
                i = feat_names.index(name)
                print(f"  {name}: delta={feat_mse_prior[i] - feat_mse_post[i]:+.4f}  (post={feat_mse_post[i]:.4f}  prior={feat_mse_prior[i]:.4f})")

        wm_metrics = {
            "step": epoch,
            "recon_error": recon_error / valid_episodes,
            "reward_nll_post": reward_nll_post / valid_episodes,
            "reward_nll_prior": reward_nll_prior / valid_episodes,
            "reward_nll": (reward_nll_post + reward_nll_prior) / valid_episodes,
            "reward_mae_post": reward_mae_post_sum / max(reward_mae_post_count, 1),
            "reward_mae_prior": reward_mae_prior_sum / max(reward_mae_prior_count, 1),
            "wm_return_mean": float(phys_episode_returns.mean()),
            "wm_return_std": float(phys_episode_returns.std()),
            "wm_return_min": float(phys_episode_returns.min()),
            "wm_return_max": float(phys_episode_returns.max()),
            "wm_return_gt_10_frac": float((phys_episode_returns > 10).mean()),
            "wm_return_lt_minus10_frac": float((phys_episode_returns < -10).mean()),
            "true_mortality": float(mortalities.mean()),
            "reward_action_range_mean": float(np.mean(reward_action_range_list)),
            "reward_action_range_std": float(np.std(reward_action_range_list)),
            "reward_action_std_mean": float(np.mean(reward_action_std_list)),
            "reward_action_std_std": float(np.std(reward_action_std_list)),
            "reward_action0_rank_mean": float(np.mean(reward_action0_rank_list)),
            "reward_action0_best_frac": float(np.mean(np.array(reward_action_best_list) == 0)),
            "cont_action0_best_frac": float(np.mean(np.array(cont_action_best_list) == 0)),
        }

        reward_best_action_counts = np.bincount(np.array(reward_action_best_list, dtype=np.int64), minlength=self._config.num_actions)
        cont_best_action_counts = np.bincount(np.array(cont_action_best_list, dtype=np.int64), minlength=self._config.num_actions)
        for a, count in enumerate(reward_best_action_counts):
            wm_metrics[f"reward_best_action_{a}_frac"] = float(count / max(len(reward_action_best_list), 1))
        for a, count in enumerate(cont_best_action_counts):
            wm_metrics[f"cont_best_action_{a}_frac"] = float(count / max(len(cont_action_best_list), 1))

        wm_metrics["cont_acc"] = cont_correct / max(cont_total, 1)

        if _nf > 0:
            wm_metrics["recon_mse_post_mean"]  = float(feat_mse_post.mean())
            wm_metrics["recon_mse_prior_mean"] = float(feat_mse_prior.mean())
            wm_metrics["recon_mse_delta_mean"] = float((feat_mse_prior - feat_mse_post).mean())
            wm_metrics["recon_mse_post_max"]   = float(feat_mse_post.max())
            wm_metrics["recon_mse_prior_max"]  = float(feat_mse_prior.max())
            wm_metrics["recon_mse_worst_post_feat"]  = feat_names[int(feat_mse_post.argmax())]
            wm_metrics["recon_mse_worst_prior_feat"] = feat_names[int(feat_mse_prior.argmax())]
            wm_metrics["recon_mse_worst_delta_feat"] = feat_names[int((feat_mse_prior - feat_mse_post).argmax())]

        if self._config.cont_type == "cont":
            wm_metrics["cont_pos_acc"] = cont_pos_correct / max(cont_pos_total, 1)
            wm_metrics["cont_neg_acc"] = cont_neg_correct / max(cont_neg_total, 1)
            wm_metrics["cont_prob_mean_on_pos"] = float(np.mean(cont_prob_on_pos)) if cont_prob_on_pos else 0.0
            wm_metrics["cont_prob_mean_on_neg"] = float(np.mean(cont_prob_on_neg)) if cont_prob_on_neg else 0.0
        elif self._config.cont_type == "mort2":
            wm_metrics["mort2_nondeath_acc"] = cont_pos_correct / max(cont_pos_total, 1)
            wm_metrics["mort2_death_acc"] = cont_neg_correct / max(cont_neg_total, 1)
            wm_metrics["mort2_prob_mean_on_nondeath"] = float(np.mean(cont_prob_on_pos)) if cont_prob_on_pos else 0.0
            wm_metrics["mort2_prob_mean_on_death"] = float(np.mean(cont_prob_on_neg)) if cont_prob_on_neg else 0.0
        elif self._config.cont_type == "mort3":
            for cls in [0, 1, 2]:
                wm_metrics[f"cont_class_{cls}_acc"] = cont_class_correct[cls] / max(cont_class_total[cls], 1)
                wm_metrics[f"cont_class_{cls}_prob_mean"] = float(np.mean(cont_pred_class_probs[cls])) if cont_pred_class_probs[cls] else 0.0
            wm_metrics["mort3_terminal_acc"] = mort3_terminal_correct / max(mort3_terminal_total, 1)
            wm_metrics["mort3_nonterminal_acc"] = mort3_nonterminal_correct / max(mort3_nonterminal_total, 1)
            wm_metrics["mort3_death_terminal_acc"] = mort3_death_terminal_correct / max(mort3_death_terminal_total, 1)
            wm_metrics["mort3_survival_terminal_acc"] = mort3_survival_terminal_correct / max(mort3_survival_terminal_total, 1)

        np.savez(
            os.path.join(self._logdir, f"phys_and_mortality_{epoch}.npz"),
            phys=phys_episode_returns, mort=mortalities,
        )
        with open(os.path.join(self._logdir, f"result_dict_{epoch}.pkl"), "wb") as f:
            pickle.dump(features_dict, f)
        with open(os.path.join(self._logdir, f"wm_eval_metrics_{epoch}.json"), "w") as f:
            json.dump(wm_metrics, f, indent=2)

        print("\n[WM EVAL SUMMARY]")
        for k, v in wm_metrics.items():
            print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

        self._update_metrics({k: v for k, v in wm_metrics.items() if k != "step"})

    def _eval(self, episodes, epoch=None):
        metrics = {}
        images = {}
        valid_episodes = 0
        cont_acc = 0.0
        recon_error = 0.0
        reward_nll_post = 0.0
        reward_nll_prior = 0.0
        true_mortality = 0
        imag_rewards = 0.0

        phys_episode_returns = []
        ai_episode_returns = []
        value_estimates = []
        ai_actions = []
        mortalities = []
        ope_trajs = []
        ai_sample_counts = np.zeros(self._config.num_actions, dtype=np.int64)

        self._set_eval_mode()
        first_debug_done = False

        with torch.no_grad():
            for stay_id, data in tqdm(episodes.items()):
                data = self._expand_episode(data)
                B, T, _ = data["features"].shape

                debug_now = not first_debug_done
                post, embed, data = self._wm._load(data, debug=debug_now, debug_name=f"_eval stay={stay_id}")

                phys_action = data["action"].detach().clone()
                is_first = data["is_first"].detach().clone()

                if phys_action.shape[1] <= 5:
                    print(f"Skipping short episode {stay_id} with length {phys_action.shape[1]}", flush=True)
                    continue

                valid_episodes += 1

                full_states, _ = self._wm.dynamics.observe(embed, phys_action, is_first, debug=False)
                pi_b_seq = self._compute_pi_b_seq(full_states, phys_action, data) if self._behavior_policy_loaded else None

                states = {k: v[:, :5] for k, v in full_states.items()}
                feat_post = self._wm.dynamics.get_feat(states)
                init = {k: v[:, 4] for k, v in full_states.items()}

                feat_init = self._wm.dynamics.get_feat(init)
                value_estimates.append(float(tools.to_np(self._get_policy_value_estimate(feat_init.detach()).squeeze())))

                self._wm.dynamics._debug_img_mode = debug_now
                if debug_now:
                    self._wm.dynamics._debug_img_counter = 0
                prior = self._wm.dynamics.imagine_with_action(phys_action[:, 5:], init)
                self._wm.dynamics._debug_img_mode = False

                feat_prior = self._wm.dynamics.get_feat(prior)

                recon = self._wm.heads["decoder"](feat_post)["features"].mode()
                openl = self._wm.heads["decoder"](feat_prior)["features"].mode()
                reward_head_post = self._wm.heads["reward"](feat_post)
                reward_head_prior = self._wm.heads["reward"](feat_prior)
                reward_prior = reward_head_prior.mode()
                cont_post = self._wm.heads["cont"](feat_post).mode()
                cont_prior = self._wm.heads["cont"](feat_prior).mode()

                model = torch.cat([recon[:, :5], openl], 1)
                comb_cont = torch.cat([cont_post[:, :5], cont_prior], 1)

                error = ((model - data["features"]) ** 2) * data["mask"]
                recon_error += (error.sum(dim=-1) / (data["mask"].sum(dim=-1) + 1e-8)).mean().item()
                reward_nll_post += (-reward_head_post.log_prob(data["reward"][:, :5])).mean().item()
                reward_nll_prior += (-reward_head_prior.log_prob(data["reward"][:, 5:])).mean().item()
                cont_acc += tools.compute_accuracy(comb_cont, data["cont"])

                imag_feat, imag_state, imag_action = self._task_behavior._imagine_in_time(
                    init, self._task_behavior.actor, 50
                )
                imag_rewards += float(tools.to_np(
                    self._wm.heads["reward"](self._wm.dynamics.get_feat(imag_state)).mode().sum()
                ))

                T_roll = prior["stoch"].shape[1]
                ai_episode_return, ai_actions_ep, ai_probs_np = self._run_ai_rollout(full_states, T_roll)
                ai_actions.append(ai_actions_ep)
                ai_sample_counts += np.bincount(ai_actions_ep, minlength=self._config.num_actions)

                if debug_now:
                    print("\n[EVAL AI POLICY PROBS DEBUG]")
                    print("mean_probs:", np.round(ai_probs_np.mean(axis=0), 4).tolist())
                    print("mean_probs_max:", float(ai_probs_np.mean(axis=0).max()))
                    print("mean_probs_argmax:", int(ai_probs_np.mean(axis=0).argmax()))

                ai_episode_returns.append(ai_episode_return)

                if self._behavior_policy_loaded:
                    pi_ai_clin_list, pi_b_clin_list, reward_list, clin_action_list = self._compute_ope_loop(
                        full_states, phys_action, pi_b_seq, data
                    )
                    traj_ope = tools.compute_ope_trajectory(
                        pi_ai_clin_list, pi_b_clin_list, reward_list,
                        gamma=self._config.discount, prob_eps=1e-6, rho_max=5.0, max_ope_steps=30,
                    )
                    if traj_ope is not None:
                        ope_trajs.append(self._augment_traj_debug(traj_ope, clin_action_list, stay_id))

                phys_episode_returns.append(tools.to_np(data["reward"][:, 5:].sum(dim=1).squeeze()))
                mortalities.append(tools.to_np(data["mortality"][:, 0].squeeze()))
                if data["mortality"].any():
                    true_mortality += 1

                if debug_now:
                    first_debug_done = True

        if valid_episodes == 0:
            print("No valid episodes for evaluation.", flush=True)
            return

        metrics["recon_error"] = recon_error / valid_episodes
        metrics["reward_nll_post"] = reward_nll_post / valid_episodes
        metrics["reward_nll_prior"] = reward_nll_prior / valid_episodes
        metrics["reward_nll"] = (reward_nll_post + reward_nll_prior) / valid_episodes
        metrics["cont_acc"] = cont_acc / valid_episodes

        ope_summary, fig, fig_value, *_ = self._finalize_ope(
            ope_trajs, ai_episode_returns, phys_episode_returns,
            mortalities, value_estimates, true_mortality,
            valid_episodes, imag_rewards, ai_sample_counts
        )
        if self._behavior_policy_loaded:
            metrics.update(ope_summary)
        images["mortality_vs_expected_return"] = fig
        images["mortality_vs_value"] = fig_value

        if epoch is not None:
            fig_value.savefig(os.path.join(self._logdir, f"mortality_vs_value_{epoch}.png"))

        self._update_metrics(metrics)
        for name, value in images.items():
            self._images.setdefault(name, []).append(value)

    def _eval_behavior(self, episodes):
        total_loss = 0.0
        total_correct = 0.0
        total_clin_prob = 0.0
        total_entropy = 0.0
        total_steps = 0

        self._set_eval_mode()

        with torch.no_grad():
            for stay_id, data in tqdm(episodes.items(), desc="Evaluating Behavior Policy"):
                data = self._expand_episode(data)
                post, embed, data = self._wm._load(data)
                if self._behavior_policy.input_type == "raw":
                    feat = data["features"]
                else:
                    feat = self._wm.dynamics.get_feat(post)

                feat_in = feat[:, :-1]
                action_tgt = data["action"][:, 1:]
                is_first_in = data["is_first"][:, :-1]

                dist = self._behavior_policy(feat_in, is_first_in) if self._behavior_policy.policy_type == "lstm" \
                    else self._behavior_policy(feat_in)

                total_loss += (-dist.log_prob(action_tgt)).sum().item()
                total_correct += (torch.argmax(dist.probs, dim=-1) == torch.argmax(action_tgt, dim=-1)).float().sum().item()
                total_clin_prob += (dist.probs * action_tgt).sum(dim=-1).sum().item()
                total_entropy += dist.entropy().sum().item()
                total_steps += action_tgt.shape[0] * action_tgt.shape[1]

        if total_steps == 0:
            print("No valid steps for behavior evaluation.", flush=True)
            return

        self._update_metrics({
            "behavior_loss_eval": total_loss / total_steps,
            "behavior_acc_eval": total_correct / total_steps,
            "behavior_clin_prob_eval": total_clin_prob / total_steps,
            "behavior_entropy_eval": total_entropy / total_steps,
        })

    def eval_behavior_policy(self, episodes, epoch):
        """Evaluate behavior policy and save per-timestep CSV for analysis."""
        rows = []
        self._set_eval_mode()

        with torch.no_grad():
            for stay_id, data in tqdm(episodes.items(), desc="Eval behavior policy"):
                data = self._expand_episode(data)
                post, embed, data = self._wm._load(data)
                if self._behavior_policy.input_type == "raw":
                    feat = data["features"]
                else:
                    feat = self._wm.dynamics.get_feat(post)

                feat_in = feat[:, :-1]
                action_tgt = data["action"][:, 1:]
                is_first_in = data["is_first"][:, :-1]

                if self._behavior_policy.policy_type == "lstm":
                    dist = self._behavior_policy(feat_in, is_first_in)
                else:
                    dist = self._behavior_policy(feat_in)

                probs = dist.probs                                             # (1, T-1, n_actions)
                clin_ids = torch.argmax(action_tgt, dim=-1)                   # (1, T-1)
                pi_b_clin = probs[0, torch.arange(probs.shape[1]), clin_ids[0]]  # (T-1,)
                bp_top1_action = torch.argmax(probs, dim=-1)[0]             # (T-1,)
                top1_hit = (bp_top1_action == clin_ids[0])
                top3 = torch.topk(probs[0], k=3, dim=-1).indices              # (T-1, 3)
                top3_hit = (top3 == clin_ids[0].unsqueeze(-1)).any(-1)
                entropy = dist.entropy()[0]                                    # (T-1,)
                mortality = float(data["mortality"][0, 0].item())

                for t in range(feat_in.shape[1]):
                    rows.append({
                        "clin_action": int(clin_ids[0, t].item()),
                        "bp_action": int(bp_top1_action[t].item()),
                        "pi_b_clin": float(pi_b_clin[t].item()),
                        "top1_hit": int(top1_hit[t].item()),
                        "top3_hit": int(top3_hit[t].item()),
                        "entropy": float(entropy[t].item()),
                        "mortality": mortality,
                    })

        out_path = os.path.join(self._logdir, f"bp_eval_{epoch}.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Saved behavior policy eval to: {out_path}", flush=True)

    def _train_wm(self, data):
        post, _, mets = self._wm._train(data)
        self._update_metrics(mets)
        return post

    def _train_policy(self, post, feat, data, use_history=True):
        if self._config.cont_type == "cont":
            reward = lambda f, s, a: self._wm.heads["reward"](self._wm.dynamics.get_feat(s)).mode()
        elif self._config.cont_type == "mort2":
            reward = lambda f, s, a: (
                self._wm.heads["reward"](self._wm.dynamics.get_feat(s)).mode()
                - 0.1 * (1.0 - self._wm.heads["cont"](self._wm.dynamics.get_feat(s)).mode())
            )
        elif self._config.cont_type == "mort3":
            def reward(f, s, a):
                feat_s = self._wm.dynamics.get_feat(s)
                return self._wm.heads["reward"](feat_s).mode() + \
                    self._cont_penalty(self._wm.heads["cont"](feat_s).probs).unsqueeze(-1)

        if use_history:
            if self._config.p1_type == "combine":
                mets = self._task_behavior._train(post, reward, feat, data, use_history)[-1]
            elif self._config.p1_type == "replay":
                mets = self._task_behavior._train_p1(post, feat, data)[-1]
            elif self._config.p1_type == "td":
                mets = self._task_behavior._train_p1_td(feat, data)
        else:
            mets = self._task_behavior._train(post, reward, feat, data, use_history)[-1]

        self._update_metrics(mets)

    def _eval_log(self, model_name, epoch):
        if epoch >= self._config.eval_every and self._should_eval(epoch):
            if self._config.mode == "behavior":
                self._eval_behavior(self._eval_dataset)
            elif self._config.mode == "world_model":
                self.eval_wm(self._eval_dataset, epoch)
            else:
                self._eval(self._eval_dataset, epoch=epoch)

        if epoch >= self._config.log_every and self._should_log(epoch):
            for name, values in self._metrics.items():
                if values:
                    if not isinstance(values[0], str):
                        self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
            for name, values in self._images.items():
                if values:
                    self._logger.image(name, values)
                    self._images[name] = []
            self._logger.write(epoch // self._config.log_every)
            best_summary = tools.extract_best_from_json(self._logdir)
            with open(self._logdir / "best_metrics.jsonl", "w") as f:
                json.dump(best_summary, f, indent=2)

        if epoch >= self._config.save_every and self._should_save(epoch):
            tools.save_model(self, model_name, self._logdir, epoch)
            with open(self._logdir / "parameters.jsonl", "w") as f:
                json.dump(vars(self._config), f, indent=2)

    def _train_behavior(self, data):
        self._behavior_policy.train()
        self._wm.eval()
        with torch.no_grad():
            post, embed, data = self._wm._load(data)
            if self._behavior_policy.input_type == "raw":
                feat = data["features"]
            else:
                feat = self._wm.dynamics.get_feat(post)
        self._update_metrics(self._behavior_policy.train_batch(feat, data["action"], data["is_first"]))


class MedDreamer(Dreamer):
    def train_wm(self, epochs):
        for epoch in trange(0, epochs + 1, desc="Training World Model"):
            _ = self._train_wm(next(self._train_dataset))
            self._eval_log("wm", epoch)

    def train_policy(self, epochs, use_history=True):
        for epoch in trange(0, epochs + 1, desc="Training Policy"):
            states, _, data = self._wm._load(next(self._train_dataset))
            feat = self._wm.dynamics.get_feat(states)
            self._train_policy(states, feat, data, use_history)
            self._eval_log("all", epoch)

    def train_behavior(self, epochs):
        for epoch in trange(0, epochs + 1, desc="Training Behavior Policy"):
            self._train_behavior(next(self._train_dataset))
            self._eval_log("behavior_policy", epoch)
