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
        embed_dim = self._wm.embed_size
        self._behavior_policy = models.BehaviorPolicy(config, embed_dim)
    
    def train(self, epochs):  # similar to the original train function in trainer
        for epoch in trange(0, epochs + 1, desc="Training"):
            states = self._train_wm(next(self._train_dataset))
            self._train_policy_through_imagination(states)
            self._eval_log("all", epoch)
    
    def eval(self, episodes, epoch):
        metrics = {"step": epoch}
        ess = 0
        v_cwpdis = 0
        imag_rewards = 0
        true_mortality = 0
        phys_episode_returns = []
        ai_episode_returns = []
        ai_actions = []
        phys_actions = []
        sofas = []
        mortalities = []
        full_mort = []
        valid_episodes = 0
        features_dict = {"ori_feat": [], "recon_feat": []}
        
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

            states, _ = self._wm.dynamics.observe(
                    embed[:, :5], phys_action[:, :5], is_first[:, :5]
                )

            # evaluate the world model
            init = {k: v[:, -1] for k, v in states.items()}

            prior = self._wm.dynamics.imagine_with_action(phys_action[:, 5:], init)
            reward_prior = self._wm.heads["reward"](self._wm.dynamics.get_feat(prior)).mode()
            def cont_penalty(cont_pred, a = 0.01, b = 0.01, c = 0.001):
                # Penalty: died=-a, discharged=+b, ICU=-c
                weights = torch.tensor([-a, b, -c], device=cont_pred.device)
                return weights[cont_pred.argmax(-1)]
            reward = reward_prior
            # reward = reward_prior + cont_penalty(self._wm.heads["cont"](self._wm.dynamics.get_feat(prior)).mode()).unsqueeze(-1)
            phys_episode_return = to_np(reward.sum(dim=1).squeeze())
            phys_episode_returns.append(phys_episode_return)
            mortalities.append(to_np(data["mortality"][:, 0].squeeze()))

            imag_feat, imag_state, imag_action = self._task_behavior._imagine_in_time(init, self._task_behavior.actor, 50)
            imag_reward = self._wm.heads["reward"](self._wm.dynamics.get_feat(imag_state)).mode()   
            imag_rewards += float(to_np(imag_reward.sum()))

            recon = self._wm.heads["decoder"](self._wm.dynamics.get_feat(states))["features"].mode()
            openl = self._wm.heads["decoder"](self._wm.dynamics.get_feat(prior))["features"].mode()
            recon_feature = torch.cat([recon[:, :5], openl], 1)
            features_dict["ori_feat"].append(to_np(data["features"]))
            features_dict["recon_feat"].append(to_np(recon_feature))

            actions = []
            clin_probs = []
            ai_episode_return = 0
            _, T, _, _ = prior["stoch"].shape

            for t in range(T):
                init = {k: v[:, t] for k, v in prior.items()}
                feat = self._wm.dynamics.get_feat(init)
                inp = feat.detach()

                actor_dist = self._task_behavior.actor(inp)   # AI policy distribution
                action = actor_dist.sample()                  # AI sampled action
                actions.append(to_np(action))

                clin_action_onehot = phys_action[:, 5 + t]    # medician action
                logp_clin = actor_dist.log_prob(clin_action_onehot)   # log pi_AI(a_clin | s_t)
                p_clin = torch.exp(logp_clin)                 # pi_AI(a_clin | s_t)
                clin_probs.append(to_np(p_clin))

                succ = self._wm.dynamics.img_step(init, action)
                ai_value = self._wm.heads["reward"](self._wm.dynamics.get_feat(succ).detach()).mode()
                ai_episode_return += to_np(ai_value.squeeze())

            actions = np.stack(actions, axis=1)
            clin_probs = np.stack(clin_probs, axis=1)
            
            ai_actions.append(np.argmax(np.squeeze(actions, axis=0), axis=-1))
            phys_actions.append(np.argmax(to_np(phys_action[0, 5:]), axis=-1))
            sofas.append(to_np(data["sofa"][0, 5:]))
            full_mort.append(to_np(data["mortality"][0, 5:]))

            ai_episode_returns.append(ai_episode_return)

            v_cwpdis_per_traj, ess_per_traj = tools.cwpdis_ess_eval(
                clin_probs, data["reward"][:, 5:], gamma=0.99
            )
            v_cwpdis += v_cwpdis_per_traj
            ess += ess_per_traj

            if data["mortality"].any() == 1:
                true_mortality += 1

        if valid_episodes == 0:
            print("No valid episodes for evaluation.", flush=True)
            return

        phys_episode_returns = np.array(phys_episode_returns, dtype=np.float32)
        ai_episode_returns = np.array(ai_episode_returns, dtype=np.float32)
        ai_actions = np.concatenate(ai_actions, axis=0)
        phys_actions = np.concatenate(phys_actions, axis=0)
        sofas = np.concatenate(sofas, axis=0)
        full_mort = np.concatenate(full_mort, axis=0)
        mortalities = np.array(mortalities, dtype=np.float32)
        fig, bin_centers, smoothed, smoothed_sem = tools.plot_mortality_vs_expected_return(phys_episode_returns, mortalities)
        fig.savefig(os.path.join(self._logdir, f"mortality_vs_expected_return_{epoch}.png"))

        ai_mortality, ai_std = tools.calculate_esmitated_mortality(ai_episode_returns, bin_centers, smoothed, smoothed_sem)
        true_mortality = true_mortality / valid_episodes
        mortality_decrease = true_mortality - ai_mortality
        metrics["mortality_decrease"] = round(mortality_decrease * 100, 2)
        metrics["ai_mortality"] = round(ai_mortality * 100, 2)
        metrics["true_mortality"] = true_mortality
        metrics["v_cwpdis"] = to_np(v_cwpdis)/ valid_episodes
        metrics["ess"] = to_np(ess)/ valid_episodes
        metrics["imag_episode_return"] = to_np(imag_rewards) / valid_episodes
        metrics["ai_episode_return"] = float(ai_episode_returns.mean())

        data = {
            "mortality": full_mort,
            "phys_action": phys_actions,
            "ai_action": ai_actions,
            "sofa": sofas
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self._logdir, f"result_data_{epoch}.csv"), index=False) # results for each transition -> action distribution plot

        np.savez(os.path.join(self._logdir, f"phys_and_mortality_{epoch}.npz"), # episode return and mortality per trajectory -> mortality vs return plot
                phys=phys_episode_returns, 
                mort=mortalities)

        with (self._logdir / f"metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n") # OPE metrics and estimated mortality

    def eval_wm(self, episodes, epoch):
        phys_episode_returns = []
        ai_episode_returns = []
        mortalities = []
        features_dict = {"ori_feat": [], "recon_feat": []}
        
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
            states, _ = self._wm.dynamics.observe(
                    embed[:, :5], phys_action[:, :5], is_first[:, :5]
                )

            # evaluate the world model
            init = {k: v[:, -1] for k, v in states.items()}

            if phys_action.shape[1] <= 5:
                print(f"Skipping short episode {stay_id} with length {phys_action.shape[1]}", flush=True)
                continue

            prior = self._wm.dynamics.imagine_with_action(phys_action[:, 5:], init)
            reward_prior = self._wm.heads["reward"](self._wm.dynamics.get_feat(prior)).mode()
            def cont_penalty(cont_pred, a = 0.01, b = 0.01, c = 0.001):
                # Penalty: died=-a, discharged=+b, ICU=-c
                weights = torch.tensor([-a, b, -c], device=cont_pred.device)
                return weights[cont_pred.argmax(-1)]
            reward = reward_prior
            # reward = reward_prior + cont_penalty(self._wm.heads["cont"](self._wm.dynamics.get_feat(prior)).mode()).unsqueeze(-1)
            phys_episode_return = to_np(reward.sum(dim=1).squeeze())
            phys_episode_returns.append(phys_episode_return)
            mortalities.append(to_np(data["mortality"][:, 0].squeeze()))

            recon = self._wm.heads["decoder"](self._wm.dynamics.get_feat(states))["features"].mode()
            openl = self._wm.heads["decoder"](self._wm.dynamics.get_feat(prior))["features"].mode()
            recon_feature = torch.cat([recon[:, :5], openl], 1)
            features_dict["ori_feat"].append(to_np(data["features"]))
            features_dict["recon_feat"].append(to_np(recon_feature))

        phys_episode_returns = np.array(phys_episode_returns, dtype=np.float32)
        ai_episode_returns = np.array(ai_episode_returns, dtype=np.float32)
        mortalities = np.array(mortalities, dtype=np.float32)
        fig, bin_centers, smoothed, smoothed_sem = tools.plot_mortality_vs_expected_return(phys_episode_returns, mortalities)
        fig.savefig(os.path.join(self._logdir, f"mortality_vs_expected_return_{epoch}.png"))

        np.savez(os.path.join(self._logdir, f"phys_and_mortality_{epoch}.npz"), 
                phys=phys_episode_returns, 
                mort=mortalities)

        with open(os.path.join(self._logdir, f"result_dict_{epoch}.pkl"), 'wb') as f:
            pickle.dump(features_dict, f)

    def _eval(self, episodes): 
        # to check if the world model is imagining correctly and reconstructing the image correctly
        metrics = {}
        images = {}
        valid_episodes = 0
        cont_acc = 0
        recon_error = 0
        reward_error = 0
        ess = 0
        v_cwpdis = 0
        phys_episode_returns = []
        ai_episode_returns = []
        ai_actions = []
        mortalities = []
        true_mortality = 0
        imag_rewards = 0
        self._wm.eval()
        
        for stay_id, data in episodes.items():
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
            states, _ = self._wm.dynamics.observe(
                    embed[:, :5], phys_action[:, :5], is_first[:, :5]
                )

            # evaluate the world model
            recon = self._wm.heads["decoder"](self._wm.dynamics.get_feat(states))["features"].mode()
            reward_post = self._wm.heads["reward"](self._wm.dynamics.get_feat(states)).mode()
            cont_post = self._wm.heads["cont"](self._wm.dynamics.get_feat(states)).mode()
            init = {k: v[:, -1] for k, v in states.items()}

            if phys_action.shape[1] <= 5:
                print(f"Skipping short episode {stay_id} with length {phys_action.shape[1]}", flush=True)
                continue

            valid_episodes += 1

            prior = self._wm.dynamics.imagine_with_action(phys_action[:, 5:], init)
            openl = self._wm.heads["decoder"](self._wm.dynamics.get_feat(prior))["features"].mode()
            reward_prior = self._wm.heads["reward"](self._wm.dynamics.get_feat(prior)).mode()
            phys_episode_return = to_np(reward_prior.sum(dim=1).squeeze())
            cont_prior = self._wm.heads["cont"](self._wm.dynamics.get_feat(prior)).mode()
            # observation is given until 5 steps
            model = torch.cat([recon[:, :5], openl], 1)
            comb_reward = torch.cat([reward_post[:, :5], reward_prior], 1)
            comb_cont = torch.cat([cont_post[:, :5], cont_prior], 1)
            error = ((model - data["features"]) ** 2) * data["mask"]
            recon_error += (error.sum(dim=-1) / (data["mask"].sum(dim=-1) + 1e-8)).mean()
            reward_error += ((comb_reward - data["reward"]) ** 2).mean()
            cont_acc += tools.compute_accuracy(comb_cont, data["cont"])
            
            if self._config.mode != "world_model":
                # evaluate rollouts
                imag_feat, imag_state, imag_action = self._task_behavior._imagine_in_time(init, self._task_behavior.actor, 50)
                imag_reward = self._wm.heads["reward"](self._wm.dynamics.get_feat(imag_state)).mode()   
                imag_rewards += float(to_np(imag_reward.sum()))

                # only evaluate one transition otherwise it's not a trajectory and cannot be sent to critic for value calculation
                actions = []
                clin_probs = []
                ai_episode_return = 0
                _, T, _, _ = prior["stoch"].shape

                for t in range(T):
                    init = {k: v[:, t] for k, v in prior.items()}
                    feat = self._wm.dynamics.get_feat(init)
                    inp = feat.detach()

                    actor_dist = self._task_behavior.actor(inp)   # AI policy distribution
                    action = actor_dist.sample()                  # AI sampled action
                    actions.append(to_np(action))

                    clin_action_onehot = phys_action[:, 5 + t]    # medician action
                    logp_clin = actor_dist.log_prob(clin_action_onehot)   # log pi_AI(a_clin | s_t)
                    p_clin = torch.exp(logp_clin)                 # pi_AI(a_clin | s_t)
                    clin_probs.append(to_np(p_clin))

                    succ = self._wm.dynamics.img_step(init, action)
                    ai_value = self._wm.heads["reward"](self._wm.dynamics.get_feat(succ).detach()).mode()
                    ai_episode_return += to_np(ai_value.squeeze())

                actions = np.stack(actions, axis=1)
                clin_probs = np.stack(clin_probs, axis=1)
                
                ai_actions.append(np.argmax(np.squeeze(actions, axis=0), axis=-1))
                ai_episode_returns.append(ai_episode_return)

                v_cwpdis_per_traj, ess_per_traj = tools.cwpdis_ess_eval(
                    clin_probs, data["reward"][:, 5:], gamma=0.99
                )
                v_cwpdis += v_cwpdis_per_traj
                ess += ess_per_traj

                phys_episode_returns.append(phys_episode_return)
                mortalities.append(to_np(data["mortality"][:, 0].squeeze()))
                if data["mortality"].any() == 1:
                    true_mortality += 1   

        if valid_episodes == 0:
            print("No valid episodes for evaluation.", flush=True)
            return

        metrics["recon_error"] = to_np(recon_error) / valid_episodes
        metrics["reward_error"] = to_np(reward_error) / valid_episodes
        metrics["cont_acc"] = to_np(cont_acc) / valid_episodes

        if self._config.mode != "world_model":
            phys_episode_returns = np.array(phys_episode_returns, dtype=np.float32)
            ai_episode_returns = np.array(ai_episode_returns, dtype=np.float32)
            mortalities = np.array(mortalities, dtype=np.float32)
            ai_actions = np.concatenate(ai_actions, axis=0)

            fig, bin_centers, smoothed, smoothed_sem = tools.plot_mortality_vs_expected_return(
                phys_episode_returns, mortalities
            )
            images["mortality_vs_expected_return"] = fig

            ai_mortality, ai_std = tools.calculate_esmitated_mortality(
                ai_episode_returns, bin_centers, smoothed, smoothed_sem
            )
            true_mortality = true_mortality / valid_episodes
            mortality_decrease = true_mortality - ai_mortality

            metrics["ai_mortality"] = round(ai_mortality * 100, 2)
            metrics["true_mortality"] = true_mortality
            metrics["mortality_decrease"] = round(mortality_decrease * 100, 2)
            metrics["imag_episode_return"] = imag_rewards / valid_episodes
            metrics["ai_episode_return"] = float(ai_episode_returns.mean())
            metrics["v_cwpdis"] = float(v_cwpdis) / valid_episodes
            metrics["ess"] = float(ess) / valid_episodes
            metrics["ai_action_min"] = ai_actions.min()
            metrics["ai_action_max"] = ai_actions.max()
            metrics["ai_action_mean"] = ai_actions.mean()
        
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)
        for name, value in images.items():
            if not name in self._images.keys():
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
                action = data["action"]   # [B, T, A]

                if embed.shape[1] == 0:
                    continue

                dist = self._behavior_policy(embed)   # sequence input

                loss = -dist.log_prob(action)         # [B, T]
                total_loss += loss.sum().item()

                pred_idx = torch.argmax(dist.probs, dim=-1)   # [B, T]
                true_idx = torch.argmax(action, dim=-1)       # [B, T]
                total_correct += (pred_idx == true_idx).sum().item()

                clin_prob = (dist.probs * action).sum(dim=-1) # [B, T]
                total_clin_prob += clin_prob.sum().item()

                entropy = dist.entropy()                      # [B, T]
                total_entropy += entropy.sum().item()

                total_steps += action.shape[0] * action.shape[1]

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

        post, embed, data = self._wm._load(data)
        # embed: [B, T, D_embed]

        mets = self._behavior_policy.train_batch(embed, data["action"])
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