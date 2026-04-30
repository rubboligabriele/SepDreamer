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
import torch.nn.functional as F

to_np = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)

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

    def _get_policy_value_estimate(self, feat):
        """
        feat: [B, D]
        returns: [B]
        """
        value_dist = self._task_behavior.value(feat)
        value = value_dist.mode()
        return value.squeeze(-1)
    
    def train(self, epochs):  # similar to the original train function in trainer
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
        features_dict = {"ori_feat": [], "recon_feat": []}
        ope_trajs = []

        self._wm.eval()
        self._task_behavior.eval()
        self._behavior_policy.eval()

        first_debug_done = False

        with torch.no_grad():
            for stay_id, data in tqdm(episodes.items()):
                data = {k: np.expand_dims(v, axis=0) for k, v in data.items()}
                B, T, _ = data["features"].shape
                data = self._wm.preprocess(data)
                flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
                unflatten = lambda x: x.reshape([B, T] + list(x.shape[1:]))

                features = flatten(data["features"])
                if self._config.fm["use_fm"]:
                    delta = flatten(data["delta"])
                else:
                    delta = None

                embed = self._wm.encoder(features, delta)
                embed = unflatten(embed)

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
                    print(f"{'='*80}")

                    max_t = min(12, phys_action.shape[1])
                    print("\n[FIRST STAY ALIGNMENT TABLE]")
                    for t in range(max_t):
                        a_idx = int(torch.argmax(phys_action[0, t]).item())
                        r_val = float(data["reward"][0, t].item())
                        mort = float(data["mortality"][0, t].item())
                        print(
                            f"t={t:02d} "
                            f"action_idx={a_idx:02d} "
                            f"reward={r_val:8.4f} "
                            f"mortality={mort:.0f}"
                        )

                if debug_now:
                    self._wm.dynamics._debug_mode = True
                    self._wm.dynamics._debug_obs_counter = 0
                else:
                    self._wm.dynamics._debug_mode = False

                full_states, _ = self._wm.dynamics.observe(embed, phys_action, is_first, debug=debug_now)
                feat_seq = self._wm.dynamics.get_feat(full_states)[:, :-1]   # s_t
                act_seq = phys_action[:, 1:]                                 # a_{t+1}

                is_first_seq = data["is_first"][:, :-1]
                dist_b_seq = self._behavior_policy(feat_seq, is_first_seq)
                logp_b_seq = dist_b_seq.log_prob(act_seq)                    # [B, T-1]
                pi_b_seq = torch.exp(logp_b_seq)

                self._wm.dynamics._debug_mode = False

                states = {k: v[:, :5] for k, v in full_states.items()}
                init = {k: v[:, 4] for k, v in full_states.items()}

                feat_init = self._wm.dynamics.get_feat(init)
                value_pred = self._get_policy_value_estimate(feat_init.detach())
                value_estimates.append(float(to_np(value_pred.squeeze())))

                if debug_now:
                    print("\n[ROLLOUT START]")
                    print("init state taken from t=4")
                    print("first rollout action ids from phys_action[:, 5:]")
                    max_roll = min(8, phys_action[:, 5:].shape[1])
                    for i in range(max_roll):
                        a_idx = int(torch.argmax(phys_action[0, 5 + i]).item())
                        print(f"roll_step={i:02d} uses phys_action at t={5+i:02d} -> action_idx={a_idx:02d}")

                    self._wm.dynamics._debug_img_mode = True
                    self._wm.dynamics._debug_img_counter = 0
                else:
                    self._wm.dynamics._debug_img_mode = False

                prior = self._wm.dynamics.imagine_with_action(phys_action[:, 5:], init)
                self._wm.dynamics._debug_img_mode = False

                reward_prior = self._wm.heads["reward"](self._wm.dynamics.get_feat(prior)).mode()
                reward = reward_prior

                phys_episode_return = to_np(reward.sum(dim=1).squeeze())
                phys_episode_returns.append(phys_episode_return)
                mortalities.append(to_np(data["mortality"][:, 0].squeeze()))

                imag_feat, imag_state, imag_action = self._task_behavior._imagine_in_time(
                    init, self._task_behavior.actor, 50
                )
                imag_reward = self._wm.heads["reward"](self._wm.dynamics.get_feat(imag_state)).mode()
                imag_rewards += float(to_np(imag_reward.sum()))

                recon = self._wm.heads["decoder"](self._wm.dynamics.get_feat(states))["features"].mode()
                openl = self._wm.heads["decoder"](self._wm.dynamics.get_feat(prior))["features"].mode()
                recon_feature = torch.cat([recon[:, :5], openl], 1)
                features_dict["ori_feat"].append(to_np(data["features"]))
                features_dict["recon_feat"].append(to_np(recon_feature))

                actions = []
                ai_episode_return = 0
                pi_ai_clin_list = []
                pi_b_clin_list = []
                reward_list = []
                clin_action_list = []

                T_roll = prior["stoch"].shape[1]

                if debug_now:
                    print("\n[OPE ALIGNMENT TABLE]")
                for t in range(5, phys_action.shape[1] - 1):
                    real_state_t = {k: v[:, t] for k, v in full_states.items()}
                    feat_real = self._wm.dynamics.get_feat(real_state_t)
                    inp_real = feat_real.detach()

                    clin_action_onehot = phys_action[:, t + 1]
                    clin_action_idx = int(torch.argmax(clin_action_onehot[0]).item())
                    clin_action_list.append(clin_action_idx)

                    ai_dist_real = self._task_behavior.actor(inp_real)
                    logp_ai = ai_dist_real.log_prob(clin_action_onehot)
                    pi_ai = torch.exp(logp_ai)

                    pi_b = pi_b_seq[:, t]

                    if debug_now and t < 12:
                        a_curr = int(torch.argmax(phys_action[0, t]).item())
                        a_next = int(torch.argmax(phys_action[0, t + 1]).item())
                        r_next = float(data["reward"][0, t + 1].item())
                        print(
                            f"t={t:02d} "
                            f"state=s_t, stored_action_t={a_curr:02d}, "
                            f"target_clin_action=a_(t+1)={a_next:02d}, "
                            f"reward_(t+1)={r_next:8.4f}, "
                            f"pi_ai={float(to_np(pi_ai.squeeze())):.6e}, "
                            f"pi_b={float(to_np(pi_b.squeeze())):.6e}"
                        )

                    pi_ai_clin_list.append(float(to_np(pi_ai.squeeze())))
                    pi_b_clin_list.append(float(to_np(pi_b.squeeze())))
                    reward_list.append(float(to_np(data["reward"][:, t + 1].squeeze())))

                state_ai = {k: v[:, 4] for k, v in full_states.items()}
                ai_episode_return = 0.0
                actions = []

                for t in range(T_roll):
                    feat_ai = self._wm.dynamics.get_feat(state_ai)
                    ai_dist = self._task_behavior.actor(feat_ai.detach())
                    action = ai_dist.sample()
                    actions.append(to_np(action))

                    state_ai = self._wm.dynamics.img_step(state_ai, action)
                    reward_ai = self._wm.heads["reward"](
                        self._wm.dynamics.get_feat(state_ai).detach()
                    ).mode()

                    ai_episode_return += float(to_np(reward_ai.squeeze()))

                actions = np.stack(actions, axis=1)
                ai_actions.append(np.argmax(np.squeeze(actions, axis=0), axis=-1))
                phys_actions.append(np.argmax(to_np(phys_action[0, 5:]), axis=-1))

                if "sofa" in data:
                    sofas.append(to_np(data["sofa"][0, 5:]))

                full_mort.append(to_np(data["mortality"][0, 5:]))
                ai_episode_returns.append(ai_episode_return)

                traj_ope = tools.compute_ope_trajectory(
                    pi_ai_clin_list,
                    pi_b_clin_list,
                    reward_list,
                    gamma=self._config.discount,
                    prob_eps=1e-6,
                    rho_max=5.0,
                    max_ope_steps=30,
                )
                if traj_ope is not None:

                    max_steps = min(len(clin_action_list), len(traj_ope["rho"]))
                    actions_used = np.array(clin_action_list[:max_steps], dtype=np.int64)
                    traj_ope["debug"]["stay_id"] = str(stay_id)
                    traj_ope["debug"]["actions"] = actions_used.tolist()
                    traj_ope["debug"]["frac_action_0"] = float(np.mean(actions_used == 0))
                    traj_ope["debug"]["num_action_0"] = int(np.sum(actions_used == 0))
                    traj_ope["debug"]["num_steps"] = int(len(actions_used))

                    ope_trajs.append(traj_ope)

                if data["mortality"].any() == 1:
                    true_mortality += 1

                if debug_now:
                    first_debug_done = True

        if valid_episodes == 0:
            print("No valid episodes for evaluation.", flush=True)
            return

        tools.debug_ope_summary(ope_trajs)
        ope_metrics = tools.finalize_ope(ope_trajs, debug=True, top_k=10)

        phys_episode_returns = np.array(phys_episode_returns, dtype=np.float32)
        ai_episode_returns = np.array(ai_episode_returns, dtype=np.float32)
        value_estimates = np.array(value_estimates, dtype=np.float32)
        ai_actions = np.concatenate(ai_actions, axis=0)
        phys_actions = np.concatenate(phys_actions, axis=0)

        if len(sofas) > 0:
            sofas = np.concatenate(sofas, axis=0)
        else:
            sofas = np.array([], dtype=np.float32)

        full_mort = np.concatenate(full_mort, axis=0)
        mortalities = np.array(mortalities, dtype=np.float32)

        fig, bin_centers, smoothed, smoothed_sem = tools.plot_mortality_vs_expected_return(
            phys_episode_returns, mortalities
        )
        fig.savefig(os.path.join(self._logdir, f"mortality_vs_expected_return_{epoch}.png"))

        fig_value, value_bin_centers, value_smoothed, value_smoothed_sem = tools.plot_mortality_vs_value(
            value_estimates, mortalities, xlabel="Critic Value"
        )
        fig_value.savefig(os.path.join(self._logdir, f"mortality_vs_value_{epoch}.png"))

        ai_mortality, ai_std = tools.calculate_estimated_mortality(
            ai_episode_returns, bin_centers, smoothed, smoothed_sem
        )
        true_mortality = true_mortality / valid_episodes
        mortality_decrease = true_mortality - ai_mortality

        metrics["mortality_decrease"] = round(mortality_decrease * 100, 2)
        metrics["ai_mortality"] = round(ai_mortality * 100, 2)
        metrics["true_mortality"] = round(true_mortality * 100, 2)

        metrics["wis"] = ope_metrics["wis"]
        metrics["wpdis"] = ope_metrics["wpdis"]
        metrics["cwpdis"] = ope_metrics["cwpdis"]
        metrics["ess"] = ope_metrics["ess"]

        metrics["imag_episode_return"] = to_np(imag_rewards) / valid_episodes
        metrics["ai_episode_return"] = float(ai_episode_returns.mean())
        metrics["critic_value_mean"] = float(value_estimates.mean())
        metrics["critic_value_std"] = float(value_estimates.std())

        data_out = {
            "mortality": full_mort,
            "phys_action": phys_actions,
            "ai_action": ai_actions,
        }
        if len(sofas) > 0:
            data_out["sofa"] = sofas
        df = pd.DataFrame(data_out)
        df.to_csv(os.path.join(self._logdir, f"result_data_{epoch}.csv"), index=False)

        np.savez(
            os.path.join(self._logdir, f"phys_and_mortality_{epoch}.npz"),
            phys=phys_episode_returns,
            mort=mortalities,
            value=value_estimates,
        )

        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")

    def eval_wm(self, episodes, epoch):
        phys_episode_returns = []
        mortalities = []
        features_dict = {"ori_feat": [], "recon_feat": []}

        first_debug_done = False

        # -------------------------
        # Accumulator reconstruction
        # -------------------------
        recon_error = 0.0
        valid_episodes = 0

        # -------------------------
        # Accumulator reward
        # -------------------------
        reward_nll_post = 0.0
        reward_nll_prior = 0.0

        reward_mae_post_sum = 0.0
        reward_mae_post_count = 0

        reward_mae_prior_sum = 0.0
        reward_mae_prior_count = 0

        reward_mae_terminal_sum = 0.0
        reward_mae_terminal_count = 0

        reward_mae_nonterminal_sum = 0.0
        reward_mae_nonterminal_count = 0

        terminal_sign_correct = 0
        terminal_sign_total = 0

        pred_reward_death_terminals = []
        pred_reward_survival_terminals = []
        true_reward_death_terminals = []
        true_reward_survival_terminals = []

        # -------------------------
        # Accumulator cont head
        # -------------------------
        cont_correct = 0
        cont_total = 0

        # binary case: cont / mort2
        cont_pos_correct = 0
        cont_pos_total = 0
        cont_neg_correct = 0
        cont_neg_total = 0

        cont_prob_on_pos = []
        cont_prob_on_neg = []

        # mort3 case
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
                data = {k: np.expand_dims(v, axis=0) for k, v in data.items()}
                B, T, _ = data["features"].shape
                data = self._wm.preprocess(data)

                flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
                unflatten = lambda x: x.reshape([B, T] + list(x.shape[1:]))

                features = flatten(data["features"])
                if self._config.fm["use_fm"]:
                    delta = flatten(data["delta"])
                else:
                    delta = None

                embed = self._wm.encoder(features, delta)
                embed = unflatten(embed)

                phys_action = data["action"].detach().clone()
                is_first = data["is_first"].detach().clone()

                debug_now = not first_debug_done
                if debug_now:
                    print(f"\n{'='*80}")
                    print(f"[EVAL_WM DEBUG] FIRST STAY = {stay_id}")
                    print(f"{'='*80}")

                    max_t = min(12, phys_action.shape[1])
                    print("\n[FIRST STAY ALIGNMENT TABLE]")
                    for t in range(max_t):
                        a_idx = int(torch.argmax(phys_action[0, t]).item())
                        r_val = float(data["reward"][0, t].item())
                        mort = float(data["mortality"][0, t].item())
                        term = float(data["is_terminal"][0, t].item())
                        print(
                            f"t={t:02d} "
                            f"action_idx={a_idx:02d} "
                            f"reward={r_val:8.4f} "
                            f"mortality={mort:.0f} "
                            f"is_terminal={term:.0f}",
                            flush=True,
                        )

                if phys_action.shape[1] <= 5:
                    print(
                        f"Skipping short episode {stay_id} with length {phys_action.shape[1]}",
                        flush=True,
                    )
                    continue

                valid_episodes += 1

                if debug_now:
                    self._wm.dynamics._debug_mode = True
                    self._wm.dynamics._debug_obs_counter = 0
                else:
                    self._wm.dynamics._debug_mode = False

                states, _ = self._wm.dynamics.observe(
                    embed[:, :5], phys_action[:, :5], is_first[:, :5], debug=debug_now
                )
                self._wm.dynamics._debug_mode = False

                init = {k: v[:, -1] for k, v in states.items()}

                if debug_now:
                    print("\n[ROLLOUT START FROM t=4]")
                    max_roll = min(8, phys_action[:, 5:].shape[1])
                    for i in range(max_roll):
                        a_idx = int(torch.argmax(phys_action[0, 5 + i]).item())
                        print(
                            f"roll_step={i:02d} uses phys_action at t={5+i:02d} -> action_idx={a_idx:02d}",
                            flush=True,
                        )
                    self._wm.dynamics._debug_img_mode = True
                    self._wm.dynamics._debug_img_counter = 0
                else:
                    self._wm.dynamics._debug_img_mode = False

                prior = self._wm.dynamics.imagine_with_action(phys_action[:, 5:], init)
                self._wm.dynamics._debug_img_mode = False

                feat_post = self._wm.dynamics.get_feat(states)
                feat_prior = self._wm.dynamics.get_feat(prior)

                # -------------------------
                # Heads
                # -------------------------
                reward_head_post = self._wm.heads["reward"](feat_post)
                reward_head_prior = self._wm.heads["reward"](feat_prior)

                reward_post = reward_head_post.mode()
                reward_prior = reward_head_prior.mode()

                cont_head_post = self._wm.heads["cont"](feat_post)
                cont_head_prior = self._wm.heads["cont"](feat_prior)

                recon = self._wm.heads["decoder"](feat_post)["features"].mode()
                openl = self._wm.heads["decoder"](feat_prior)["features"].mode()

                # -------------------------
                # Reconstruction
                # -------------------------
                model = torch.cat([recon[:, :5], openl], 1)
                error = ((model - data["features"]) ** 2) * data["mask"]
                recon_error += (
                    error.sum(dim=-1) / (data["mask"].sum(dim=-1) + 1e-8)
                ).mean().item()

                # -------------------------
                # Reward NLL
                # -------------------------
                reward_nll_post += (
                    -reward_head_post.log_prob(data["reward"][:, :5])
                ).mean().item()

                reward_nll_prior += (
                    -reward_head_prior.log_prob(data["reward"][:, 5:])
                ).mean().item()

                # -------------------------
                # Reward MAE
                # -------------------------
                true_reward_post = data["reward"][:, :5]
                true_reward_prior = data["reward"][:, 5:]

                mae_post = torch.abs(reward_post - true_reward_post)
                mae_prior = torch.abs(reward_prior - true_reward_prior)

                reward_mae_post_sum += mae_post.sum().item()
                reward_mae_post_count += mae_post.numel()

                reward_mae_prior_sum += mae_prior.sum().item()
                reward_mae_prior_count += mae_prior.numel()

                terminal_mask_prior = data["is_terminal"][:, 5:].bool().unsqueeze(-1)
                nonterminal_mask_prior = ~terminal_mask_prior

                if terminal_mask_prior.any():
                    term_pred = reward_prior[terminal_mask_prior]
                    term_true = true_reward_prior[terminal_mask_prior]

                    reward_mae_terminal_sum += torch.abs(term_pred - term_true).sum().item()
                    reward_mae_terminal_count += term_true.numel()

                    sign_ok = (torch.sign(term_pred) == torch.sign(term_true)).sum().item()
                    terminal_sign_correct += sign_ok
                    terminal_sign_total += term_true.numel()

                    death_mask = (
                        data["mortality"][:, 5:].bool().unsqueeze(-1) & terminal_mask_prior
                    )
                    surv_mask = (
                        (~data["mortality"][:, 5:].bool()).unsqueeze(-1) & terminal_mask_prior
                    )

                    if death_mask.any():
                        pred_reward_death_terminals.extend(
                            reward_prior[death_mask].detach().cpu().view(-1).tolist()
                        )
                        true_reward_death_terminals.extend(
                            true_reward_prior[death_mask].detach().cpu().view(-1).tolist()
                        )

                    if surv_mask.any():
                        pred_reward_survival_terminals.extend(
                            reward_prior[surv_mask].detach().cpu().view(-1).tolist()
                        )
                        true_reward_survival_terminals.extend(
                            true_reward_prior[surv_mask].detach().cpu().view(-1).tolist()
                        )

                if nonterminal_mask_prior.any():
                    nonterm_pred = reward_prior[nonterminal_mask_prior]
                    nonterm_true = true_reward_prior[nonterminal_mask_prior]
                    reward_mae_nonterminal_sum += torch.abs(nonterm_pred - nonterm_true).sum().item()
                    reward_mae_nonterminal_count += nonterm_true.numel()

                if debug_now:
                    print("\n[REWARD HEAD ALIGNMENT]")
                    max_r = min(10, reward_prior.shape[1])
                    for k in range(max_r):
                        real_t = 5 + k
                        pred_r = float(reward_prior[0, k].item())
                        real_r = float(data["reward"][0, real_t].item())
                        act_idx = int(torch.argmax(phys_action[0, real_t]).item())
                        term_flag = int(data["is_terminal"][0, real_t].item())
                        print(
                            f"roll_step={k:02d} "
                            f"uses action at t={real_t:02d} act={act_idx:02d} "
                            f"pred_reward={pred_r:8.4f} "
                            f"real_reward={real_r:8.4f} "
                            f"is_terminal={term_flag}"
                        )

                # -------------------------
                # Cont head analysis
                # -------------------------
                if self._config.cont_type in ["cont", "mort2"]:
                    cont_prob_post = cont_head_post.mean
                    cont_prob_prior = cont_head_prior.mean

                    cont_prob_all = torch.cat([cont_prob_post[:, :5], cont_prob_prior], dim=1)
                    cont_true_all = data["cont"].float()

                    cont_pred_all = (cont_prob_all >= 0.5).float()

                    cont_correct += (cont_pred_all == cont_true_all).sum().item()
                    cont_total += cont_true_all.numel()

                    pos_mask = cont_true_all == 1
                    neg_mask = cont_true_all == 0

                    if pos_mask.any():
                        cont_pos_correct += (cont_pred_all[pos_mask] == cont_true_all[pos_mask]).sum().item()
                        cont_pos_total += pos_mask.sum().item()
                        cont_prob_on_pos.extend(
                            cont_prob_all[pos_mask].detach().cpu().view(-1).tolist()
                        )

                    if neg_mask.any():
                        cont_neg_correct += (cont_pred_all[neg_mask] == cont_true_all[neg_mask]).sum().item()
                        cont_neg_total += neg_mask.sum().item()
                        cont_prob_on_neg.extend(
                            cont_prob_all[neg_mask].detach().cpu().view(-1).tolist()
                        )

                elif self._config.cont_type == "mort3":
                    cont_probs_post = cont_head_post.probs
                    cont_probs_prior = cont_head_prior.probs
                    cont_probs_all = torch.cat([cont_probs_post[:, :5], cont_probs_prior], dim=1)

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
                        mort3_death_terminal_correct += (
                            pred_cls[death_terminal_mask] == true_cls[death_terminal_mask]
                        ).sum().item()
                        mort3_death_terminal_total += death_terminal_mask.sum().item()

                    if survival_terminal_mask.any():
                        mort3_survival_terminal_correct += (
                            pred_cls[survival_terminal_mask] == true_cls[survival_terminal_mask]
                        ).sum().item()
                        mort3_survival_terminal_total += survival_terminal_mask.sum().item()

                # -------------------------
                # Returns for mortality plot
                # -------------------------
                phys_episode_return = to_np(reward_prior.sum(dim=1).squeeze())
                phys_episode_returns.append(phys_episode_return)
                mortalities.append(to_np(data["mortality"][:, 0].squeeze()))

                # -------------------------
                # Recon dumps
                # -------------------------
                recon_feature = torch.cat([recon[:, :5], openl], 1)
                features_dict["ori_feat"].append(to_np(data["features"]))
                features_dict["recon_feat"].append(to_np(recon_feature))

                if debug_now:
                    first_debug_done = True

        if valid_episodes == 0:
            print("No valid episodes for evaluation.", flush=True)
            return

        phys_episode_returns = np.array(phys_episode_returns, dtype=np.float32)
        mortalities = np.array(mortalities, dtype=np.float32)

        fig, bin_centers, smoothed, smoothed_sem = tools.plot_mortality_vs_expected_return(
            phys_episode_returns, mortalities
        )
        fig.savefig(os.path.join(self._logdir, f"mortality_vs_expected_return_{epoch}.png"))

        wm_metrics = {
            "step": epoch,
            "recon_error": recon_error / valid_episodes,
            "reward_nll_post": reward_nll_post / valid_episodes,
            "reward_nll_prior": reward_nll_prior / valid_episodes,
            "reward_mae_post": reward_mae_post_sum / max(reward_mae_post_count, 1),
            "reward_mae_prior": reward_mae_prior_sum / max(reward_mae_prior_count, 1),
            "reward_mae_terminal": reward_mae_terminal_sum / max(reward_mae_terminal_count, 1),
            "reward_mae_nonterminal": reward_mae_nonterminal_sum / max(reward_mae_nonterminal_count, 1),
            "terminal_reward_sign_acc": terminal_sign_correct / max(terminal_sign_total, 1),
            "wm_return_mean": float(phys_episode_returns.mean()),
            "wm_return_std": float(phys_episode_returns.std()),
            "wm_return_min": float(phys_episode_returns.min()),
            "wm_return_max": float(phys_episode_returns.max()),
            "wm_return_gt_10_frac": float((phys_episode_returns > 10).mean()),
            "wm_return_lt_minus10_frac": float((phys_episode_returns < -10).mean()),
            "true_mortality": float(mortalities.mean()),
        }

        if len(pred_reward_death_terminals) > 0:
            wm_metrics["pred_reward_death_terminal_mean"] = float(np.mean(pred_reward_death_terminals))
            wm_metrics["true_reward_death_terminal_mean"] = float(np.mean(true_reward_death_terminals))

        if len(pred_reward_survival_terminals) > 0:
            wm_metrics["pred_reward_survival_terminal_mean"] = float(np.mean(pred_reward_survival_terminals))
            wm_metrics["true_reward_survival_terminal_mean"] = float(np.mean(true_reward_survival_terminals))

        wm_metrics["cont_acc"] = cont_correct / max(cont_total, 1)

        if self._config.cont_type == "cont":
            wm_metrics["cont_pos_acc"] = cont_pos_correct / max(cont_pos_total, 1)
            wm_metrics["cont_neg_acc"] = cont_neg_correct / max(cont_neg_total, 1)
            wm_metrics["cont_prob_mean_on_pos"] = float(np.mean(cont_prob_on_pos)) if len(cont_prob_on_pos) > 0 else 0.0
            wm_metrics["cont_prob_mean_on_neg"] = float(np.mean(cont_prob_on_neg)) if len(cont_prob_on_neg) > 0 else 0.0

        elif self._config.cont_type == "mort2":
            wm_metrics["mort2_nondeath_acc"] = cont_pos_correct / max(cont_pos_total, 1)
            wm_metrics["mort2_death_acc"] = cont_neg_correct / max(cont_neg_total, 1)
            wm_metrics["mort2_prob_mean_on_nondeath"] = float(np.mean(cont_prob_on_pos)) if len(cont_prob_on_pos) > 0 else 0.0
            wm_metrics["mort2_prob_mean_on_death"] = float(np.mean(cont_prob_on_neg)) if len(cont_prob_on_neg) > 0 else 0.0

        elif self._config.cont_type == "mort3":
            for cls in [0, 1, 2]:
                wm_metrics[f"cont_class_{cls}_acc"] = cont_class_correct[cls] / max(cont_class_total[cls], 1)
                wm_metrics[f"cont_class_{cls}_prob_mean"] = (
                    float(np.mean(cont_pred_class_probs[cls]))
                    if len(cont_pred_class_probs[cls]) > 0 else 0.0
                )

            wm_metrics["mort3_terminal_acc"] = mort3_terminal_correct / max(mort3_terminal_total, 1)
            wm_metrics["mort3_nonterminal_acc"] = mort3_nonterminal_correct / max(mort3_nonterminal_total, 1)
            wm_metrics["mort3_death_terminal_acc"] = mort3_death_terminal_correct / max(mort3_death_terminal_total, 1)
            wm_metrics["mort3_survival_terminal_acc"] = mort3_survival_terminal_correct / max(mort3_survival_terminal_total, 1)

        np.savez(
            os.path.join(self._logdir, f"phys_and_mortality_{epoch}.npz"),
            phys=phys_episode_returns,
            mort=mortalities,
        )

        with open(os.path.join(self._logdir, f"result_dict_{epoch}.pkl"), "wb") as f:
            pickle.dump(features_dict, f)

        with open(os.path.join(self._logdir, f"wm_eval_metrics_{epoch}.json"), "w") as f:
            json.dump(wm_metrics, f, indent=2)

        print("\n[WM EVAL SUMMARY]")
        for k, v in wm_metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

    def _eval(self, episodes):
        metrics = {}
        images = {}
        valid_episodes = 0
        cont_acc = 0.0
        recon_error = 0.0
        reward_nll = 0.0
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

        self._wm.eval()
        self._task_behavior.eval()
        self._behavior_policy.eval()

        first_debug_done = False

        with torch.no_grad():
            for stay_id, data in tqdm(episodes.items()):
                data = {k: np.expand_dims(v, axis=0) for k, v in data.items()}
                B, T, _ = data["features"].shape

                debug_now = not first_debug_done
                post, embed, data = self._wm._load(
                    data,
                    debug=debug_now,
                    debug_name=f"_eval stay={stay_id}"
                )

                phys_action = data["action"].detach().clone()
                is_first = data["is_first"].detach().clone()

                full_states, _ = self._wm.dynamics.observe(embed, phys_action, is_first, debug=False)
                feat_seq = self._wm.dynamics.get_feat(full_states)[:, :-1]   # s_t
                act_seq = phys_action[:, 1:]                                 # a_{t+1}

                is_first_seq = data["is_first"][:, :-1]
                dist_b_seq = self._behavior_policy(feat_seq, is_first_seq)
                logp_b_seq = dist_b_seq.log_prob(act_seq)                    # [B, T-1]
                pi_b_seq = torch.exp(logp_b_seq)

                states = {k: v[:, :5] for k, v in full_states.items()}
                feat_post = self._wm.dynamics.get_feat(states)

                recon = self._wm.heads["decoder"](feat_post)["features"].mode()
                reward_head_post = self._wm.heads["reward"](feat_post)
                cont_post = self._wm.heads["cont"](feat_post).mode()
                init = {k: v[:, 4] for k, v in full_states.items()}

                feat_init = self._wm.dynamics.get_feat(init)
                value_pred = self._get_policy_value_estimate(feat_init.detach())
                value_estimates.append(float(to_np(value_pred.squeeze())))

                if phys_action.shape[1] <= 5:
                    print(f"Skipping short episode {stay_id} with length {phys_action.shape[1]}", flush=True)
                    continue

                valid_episodes += 1

                if debug_now:
                    self._wm.dynamics._debug_img_mode = True
                    self._wm.dynamics._debug_img_counter = 0
                else:
                    self._wm.dynamics._debug_img_mode = False

                prior = self._wm.dynamics.imagine_with_action(phys_action[:, 5:], init)
                self._wm.dynamics._debug_img_mode = False

                feat_prior = self._wm.dynamics.get_feat(prior)

                openl = self._wm.heads["decoder"](feat_prior)["features"].mode()
                reward_head_prior = self._wm.heads["reward"](feat_prior)
                reward_prior = reward_head_prior.mode()
                phys_episode_return = to_np(reward_prior.sum(dim=1).squeeze())
                cont_prior = self._wm.heads["cont"](feat_prior).mode()

                model = torch.cat([recon[:, :5], openl], 1)
                comb_cont = torch.cat([cont_post[:, :5], cont_prior], 1)

                error = ((model - data["features"]) ** 2) * data["mask"]
                recon_error += (
                    error.sum(dim=-1) / (data["mask"].sum(dim=-1) + 1e-8)
                ).mean().item()

                reward_nll_post += (
                    -reward_head_post.log_prob(data["reward"][:, :5])
                ).mean().item()

                reward_nll_prior += (
                    -reward_head_prior.log_prob(data["reward"][:, 5:])
                ).mean().item()

                reward_nll += (
                    (
                        -reward_head_post.log_prob(data["reward"][:, :5])
                    ).mean()
                    + (
                        -reward_head_prior.log_prob(data["reward"][:, 5:])
                    ).mean()
                ).item()

                cont_acc += tools.compute_accuracy(comb_cont, data["cont"])

                if self._config.mode != "world_model":
                    imag_feat, imag_state, imag_action = self._task_behavior._imagine_in_time(
                        init, self._task_behavior.actor, 50
                    )
                    imag_reward = self._wm.heads["reward"](
                        self._wm.dynamics.get_feat(imag_state)
                    ).mode()
                    imag_rewards += float(to_np(imag_reward.sum()))

                    actions = []
                    ai_episode_return = 0.0
                    pi_ai_clin_list = []
                    pi_b_clin_list = []
                    reward_list = []
                    clin_action_list = []

                    T_roll = prior["stoch"].shape[1]

                    if debug_now:
                        print("\n[_EVAL OPE ALIGNMENT TABLE]")

                    for t in range(5, phys_action.shape[1] - 1):
                        real_state_t = {k: v[:, t] for k, v in full_states.items()}
                        feat_real = self._wm.dynamics.get_feat(real_state_t)
                        inp_real = feat_real.detach()

                        clin_action_onehot = phys_action[:, t + 1]

                        clin_action_idx = int(torch.argmax(clin_action_onehot[0]).item())
                        clin_action_list.append(clin_action_idx)

                        ai_dist_real = self._task_behavior.actor(inp_real)
                        logp_ai = ai_dist_real.log_prob(clin_action_onehot)
                        pi_ai = torch.exp(logp_ai)

                        pi_b = pi_b_seq[:, t]

                        if debug_now and t < 12:
                            a_curr = int(torch.argmax(phys_action[0, t]).item())
                            a_next = int(torch.argmax(phys_action[0, t + 1]).item())
                            r_next = float(data["reward"][0, t + 1].item())
                            print(
                                f"t={t:02d} stored_action_t={a_curr:02d} "
                                f"target_action_next={a_next:02d} "
                                f"reward_next={r_next:8.4f} "
                                f"pi_ai={float(to_np(pi_ai.squeeze())):.6e} "
                                f"pi_b={float(to_np(pi_b.squeeze())):.6e}"
                            )

                        pi_ai_clin_list.append(float(to_np(pi_ai.squeeze())))
                        pi_b_clin_list.append(float(to_np(pi_b.squeeze())))
                        reward_list.append(float(to_np(data["reward"][:, t + 1].squeeze())))

                    state_ai = {k: v[:, 4] for k, v in full_states.items()}
                    ai_episode_return = 0.0
                    actions = []

                    for t in range(T_roll):
                        feat_ai = self._wm.dynamics.get_feat(state_ai)
                        ai_dist = self._task_behavior.actor(feat_ai.detach())
                        action = ai_dist.sample()
                        actions.append(to_np(action))

                        state_ai = self._wm.dynamics.img_step(state_ai, action)
                        reward_ai = self._wm.heads["reward"](
                            self._wm.dynamics.get_feat(state_ai).detach()
                        ).mode()

                        ai_episode_return += float(to_np(reward_ai.squeeze()))

                    actions = np.stack(actions, axis=1)
                    ai_actions.append(np.argmax(np.squeeze(actions, axis=0), axis=-1))
                    ai_episode_returns.append(ai_episode_return)

                    traj_ope = tools.compute_ope_trajectory(
                        pi_ai_clin_list,
                        pi_b_clin_list,
                        reward_list,
                        gamma=self._config.discount,
                        prob_eps=1e-6,
                        rho_max=5.0,
                        max_ope_steps=30,
                    )
                    if traj_ope is not None:
                        max_steps = min(len(clin_action_list), len(traj_ope["rho"]))
                        actions_used = np.array(clin_action_list[:max_steps], dtype=np.int64)

                        traj_ope["debug"]["stay_id"] = str(stay_id)
                        traj_ope["debug"]["actions"] = actions_used.tolist()
                        traj_ope["debug"]["frac_action_0"] = float(np.mean(actions_used == 0))
                        traj_ope["debug"]["num_action_0"] = int(np.sum(actions_used == 0))
                        traj_ope["debug"]["num_steps"] = int(len(actions_used))

                        ope_trajs.append(traj_ope)

                    phys_episode_returns.append(phys_episode_return)
                    mortalities.append(to_np(data["mortality"][:, 0].squeeze()))
                    if data["mortality"].any() == 1:
                        true_mortality += 1

                if debug_now:
                    first_debug_done = True

        if valid_episodes == 0:
            print("No valid episodes for evaluation.", flush=True)
            return

        metrics["recon_error"] = recon_error / valid_episodes
        metrics["reward_nll"] = reward_nll / valid_episodes
        metrics["reward_nll_post"] = reward_nll_post / valid_episodes
        metrics["reward_nll_prior"] = reward_nll_prior / valid_episodes
        metrics["cont_acc"] = cont_acc / valid_episodes

        if self._config.mode != "world_model":
            tools.debug_ope_summary(ope_trajs)
            ope_metrics = tools.finalize_ope(ope_trajs, debug=True, top_k=10)

            phys_episode_returns = np.array(phys_episode_returns, dtype=np.float32)
            ai_episode_returns = np.array(ai_episode_returns, dtype=np.float32)
            value_estimates = np.array(value_estimates, dtype=np.float32)
            mortalities = np.array(mortalities, dtype=np.float32)
            ai_actions = np.concatenate(ai_actions, axis=0)

            fig, bin_centers, smoothed, smoothed_sem = tools.plot_mortality_vs_expected_return(
                phys_episode_returns, mortalities
            )
            images["mortality_vs_expected_return"] = fig

            fig_value, value_bin_centers, value_smoothed, value_smoothed_sem = tools.plot_mortality_vs_value(
                value_estimates, mortalities, xlabel="Critic Value"
            )
            images["mortality_vs_value"] = fig_value

            ai_mortality, ai_std = tools.calculate_estimated_mortality(
                ai_episode_returns, bin_centers, smoothed, smoothed_sem
            )
            true_mortality = true_mortality / valid_episodes
            mortality_decrease = true_mortality - ai_mortality

            metrics["ai_mortality"] = round(ai_mortality * 100, 2)
            metrics["true_mortality"] = round(true_mortality * 100, 2)
            metrics["mortality_decrease"] = round(mortality_decrease * 100, 2)
            metrics["imag_episode_return"] = imag_rewards / valid_episodes
            metrics["ai_episode_return"] = float(ai_episode_returns.mean())
            metrics["critic_value_mean"] = float(value_estimates.mean())
            metrics["critic_value_std"] = float(value_estimates.std())

            metrics["wis"] = ope_metrics["wis"]
            metrics["wpdis"] = ope_metrics["wpdis"]
            metrics["cwpdis"] = ope_metrics["cwpdis"]
            metrics["ess"] = ope_metrics["ess"]

            metrics["ai_action_min"] = ai_actions.min()
            metrics["ai_action_max"] = ai_actions.max()
            metrics["ai_action_mean"] = ai_actions.mean()

        for name, value in metrics.items():
            if name not in self._metrics:
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

        for name, value in images.items():
            if name not in self._images:
                self._images[name] = [value]
            else:
                self._images[name].append(value)

    def _eval_behavior(self, episodes):
        metrics = {}

        total_loss = 0.0
        total_correct = 0.0
        total_clin_prob = 0.0
        total_entropy = 0.0
        total_steps = 0

        self._wm.eval()
        self._behavior_policy.eval()

        with torch.no_grad():
            for stay_id, data in tqdm(episodes.items(), desc="Evaluating Behavior Policy"):
                data = {k: np.expand_dims(v, axis=0) for k, v in data.items()}

                post, embed, data = self._wm._load(data)
                feat = self._wm.dynamics.get_feat(post)

                feat_in = feat[:, :-1]
                action_tgt = data["action"][:, 1:]
                is_first_in = data["is_first"][:, :-1]

                dist = self._behavior_policy(feat_in, is_first_in)

                loss = -dist.log_prob(action_tgt)
                total_loss += loss.sum().item()

                pred_idx = torch.argmax(dist.probs, dim=-1)
                true_idx = torch.argmax(action_tgt, dim=-1)

                correct = (pred_idx == true_idx).float()
                total_correct += correct.sum().item()

                clin_prob = (dist.probs * action_tgt).sum(dim=-1)
                total_clin_prob += clin_prob.sum().item()

                entropy = dist.entropy()
                total_entropy += entropy.sum().item()

                total_steps += action_tgt.shape[0] * action_tgt.shape[1]

        if total_steps == 0:
            print("No valid steps for behavior evaluation.", flush=True)
            return

        metrics["behavior_loss_eval"] = total_loss / total_steps
        metrics["behavior_acc_eval"] = total_correct / total_steps
        metrics["behavior_clin_prob_eval"] = total_clin_prob / total_steps
        metrics["behavior_entropy_eval"] = total_entropy / total_steps

        for name, value in metrics.items():
            if name not in self._metrics:
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def _train_wm(self, data): # similar to the original train_epoch function in trainer
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)
        return post
    
    def _train_policy(self, post, feat, data, use_history=True):
        metrics = {}
        start = post
        if self._config.cont_type == 'cont':
            reward = lambda f, s, a: self._wm.heads["reward"](
                self._wm.dynamics.get_feat(s)
            ).mode()
        elif self._config.cont_type == 'mort2':
            reward = lambda f, s, a: (
            self._wm.heads["reward"](self._wm.dynamics.get_feat(s)).mode()
            - 0.1 * (1.0 - self._wm.heads["cont"](self._wm.dynamics.get_feat(s)).mode())
            )
        elif self._config.cont_type == 'mort3':
            def cont_penalty(cont_pred, a = 0.01, b = 0.01, c = 0.001):
                # Penalty: died=-a, discharged=+b, ICU=-c
                weights = torch.tensor([-a, b, -c], device=cont_pred.device)
                return weights[cont_pred.argmax(-1)]
            
            reward = lambda f, s, a: (
            self._wm.heads["reward"](self._wm.dynamics.get_feat(s)).mode() +
            cont_penalty(self._wm.heads["cont"](self._wm.dynamics.get_feat(s)).mode()).unsqueeze(-1))

        if use_history:
            if self._config.p1_type == "combine":
                metrics.update(self._task_behavior._train(start, reward, feat, data, use_history)[-1])
            elif self._config.p1_type == "replay":
                metrics.update(self._task_behavior._train_p1(start, feat, data)[-1])
            elif self._config.p1_type == "td":
                metrics.update(self._task_behavior._train_p1_td(feat, data))
        else:
            metrics.update(self._task_behavior._train(start, reward, feat, data, use_history)[-1])
        
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)
    
    def _eval_log(self, model_name, epoch):
        if epoch >= self._config.eval_every and self._should_eval(epoch):
            if self._config.mode != "behavior":
                self._eval(self._eval_dataset)
            else:
                self._eval_behavior(self._eval_dataset)

        if epoch >= self._config.log_every and self._should_log(epoch):
            for name, values in self._metrics.items():
                if len(values) == 0:
                    continue
                self._logger.scalar(name, float(np.mean(values)))
                self._metrics[name] = []
            for name, values in self._images.items():
                if len(values) == 0:
                    continue
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
        metrics = {}

        self._behavior_policy.train()
        self._wm.eval()

        with torch.no_grad():
            post, embed, data = self._wm._load(data)
            feat = self._wm.dynamics.get_feat(post)

        mets = self._behavior_policy.train_batch(feat, data["action"], data["is_first"])
        metrics.update(mets)

        for name, value in metrics.items():
            if name not in self._metrics:
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

class MedDreamer(Dreamer):
    def __init__(self, config, logger, logdir, train_dataset, eval_dataset):
        super(MedDreamer, self).__init__(config, logger, logdir, train_dataset, eval_dataset)
        
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