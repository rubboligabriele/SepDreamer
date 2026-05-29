import copy
import torch
from torch import nn
import torch.nn.functional as F

import src.meddreamer.models.networks as networks
import src.meddreamer.utils.tools as tools
from src.meddreamer.models.afi import AFIEmbedding

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-1):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, config):
        super(WorldModel, self).__init__()
        self._config = config
        shapes = {"features": tuple((config.num_features,))}
        # shapes = {"features": tuple((config.num_features,)), "delta": tuple((1,))} #ablation: add delta to input
        if config.fm["use_fm"]:
            self.encoder = AFIEmbedding(config.num_features, config.fm["fm_units"])
        else:
            self.encoder = networks.MultiEncoder(shapes, **config.encoder)
       
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (3,) if self._config.cont_type == "mort3" else (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.cont_head["dist"],
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            features=config.cont_head["loss_scale"],
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        # action (batch_size, batch_length, num_actions)
        # features (batch_size, batch_length, num_features)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        B, T, _ = data["features"].shape
        data = self.preprocess(data)

        if getattr(self._config, "debug", False):
            bad_transitions = []

            for b in range(B):
                for t in range(T - 1):
                    if data["is_terminal"][b, t].item() == 1:
                        next_is_first = data["is_first"][b, t + 1].item()
                        next_reward = data["reward"][b, t + 1].item()
                        next_action = int(torch.argmax(data["action"][b, t + 1]).item())

                        if next_is_first != 1:
                            bad_transitions.append(
                                (b, t, next_is_first, next_reward, next_action)
                            )

            if bad_transitions:
                print("\n[LEAKAGE WARNING] terminal followed by non-reset!", flush=True)
                for b, t, next_is_first, next_reward, next_action in bad_transitions[:20]:
                    print(
                        f"batch={b} terminal_t={t} "
                        f"next_t={t+1} is_first={next_is_first} "
                        f"next_reward={next_reward:.4f} "
                        f"next_action={next_action}",
                        flush=True,
                    )
            else:
                print(
                    "[LEAKAGE CHECK] OK: every terminal is followed by is_first=1 or sequence ends",
                    flush=True,
                )

            terminal_positions = data["is_terminal"].nonzero(as_tuple=False)
            if terminal_positions.numel() > 0:
                terminal_ts = terminal_positions[:, 1].detach().cpu().numpy()
                print(
                    f"[TERMINAL POSITIONS] "
                    f"n={len(terminal_ts)}, "
                    f"min_t={terminal_ts.min()}, "
                    f"max_t={terminal_ts.max()}, "
                    f"mean_t={terminal_ts.mean():.2f}, "
                    f"frac_at_last={(terminal_ts == T - 1).mean():.3f}",
                    flush=True,
                )

        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        unflatten = lambda x: x.reshape([B, T] + list(x.shape[1:]))
        features = flatten(data["features"])# (batch_size, batch_length, num_features) -> (batch_size * batch_length, num_features)
        if self._config.fm["use_fm"]:
            delta = flatten(data["delta"]) 
        else:
            delta = None

        with tools.RequiresGrad(self):
            embed = self.encoder(features, delta)
            embed = unflatten(embed)# (batch_size * batch_length, num_features) -> (batch_size, batch_length, num_features)
            post, prior = self.dynamics.observe(
                embed, data["action"], data["is_first"]
            )
            kl_free = self._config.kl_free
            dyn_scale = self._config.dyn_scale
            rep_scale = self._config.rep_scale
            kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                post, prior, kl_free, dyn_scale, rep_scale
            )
            assert kl_loss.shape == embed.shape[:2], kl_loss.shape
            preds = {}
            for name, head in self.heads.items():
                grad_head = name in self._config.grad_heads
                feat = self.dynamics.get_feat(post)
                feat = feat if grad_head else feat.detach()
                pred = head(feat)
                if type(pred) is dict:
                    preds.update(pred)
                else:
                    preds[name] = pred
            losses = {}
            for name, pred in preds.items():
                loss = -pred.log_prob(data[name])
                if name == "features":
                    loss = loss * data["mask"]
                    loss = loss.sum(dim=-1) / (data["mask"].sum(dim=-1) + 1e-8)
                elif name == "reward":
                    terminal_weight = getattr(self._config, "terminal_reward_loss_weight", 20.0)
                    terminal_mask = data["is_terminal"].float()
                    weight = 1.0 + terminal_weight * terminal_mask
                    loss = loss * weight
                elif name == "cont":
                    terminal_cont_weight = getattr(self._config, "terminal_cont_loss_weight", 20.0)
                    terminal_mask = data["is_terminal"].float()
                    weight = 1.0 + terminal_cont_weight * terminal_mask
                    loss = loss * weight
                assert loss.shape == embed.shape[:2], (name, loss.shape)
                losses[name] = loss
            scaled = {
                key: value * self._scales.get(key, 1.0)
                for key, value in losses.items()
            }
            model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        # metrics["kl_free"] = kl_free
        # metrics["dyn_scale"] = dyn_scale
        # metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        metrics["prior_ent"] = to_np(
            torch.mean(self.dynamics.get_dist(prior).entropy())
        )
        metrics["post_ent"] = to_np(
            torch.mean(self.dynamics.get_dist(post).entropy())
        )
        context = dict(
            embed=embed,
            feat=self.dynamics.get_feat(post),
            kl=kl_value,
            postent=self.dynamics.get_dist(post).entropy(),
        )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics
    
    def _load(self, data, debug=False, debug_name=""):
        B, T, _ = data["features"].shape
        data = self.preprocess(data)

        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        unflatten = lambda x: x.reshape([B, T] + list(x.shape[1:]))

        features = flatten(data["features"])
        if self._config.fm["use_fm"]:
            delta = flatten(data["delta"])
        else:
            delta = None

        embed = self.encoder(features, delta)
        embed = unflatten(embed)

        action = data["action"].detach().clone()
        is_first = data["is_first"].detach().clone()

        if debug:
            print(f"\n[LOAD DEBUG] {debug_name}")
            print("features shape:", tuple(data["features"].shape))
            print("embed shape:", tuple(embed.shape))
            print("action shape:", tuple(action.shape))
            print("reward shape:", tuple(data["reward"].shape))
            print("is_first shape:", tuple(is_first.shape))
            print("delta shape:", tuple(data["delta"].shape) if "delta" in data else None)

            max_t = min(10, T)
            for t in range(max_t):
                a_idx = int(torch.argmax(action[0, t]).item())
                r_val = float(data["reward"][0, t].item())
                first_val = float(is_first[0, t].item())
                print(
                    f"t={t:02d} "
                    f"action_idx={a_idx:02d} "
                    f"reward={r_val:8.4f} "
                    f"is_first={first_val:.0f} "
                    f"embed_norm={float(embed[0, t].norm().item()):.4f}"
                )

        # attiva debug interno RSSM solo se richiesto
        if debug:
            self.dynamics._debug_mode = True
            self.dynamics._debug_obs_counter = 0
        else:
            self.dynamics._debug_mode = False
            self.dynamics._debug_obs_counter = 0

        post, prior = self.dynamics.observe(embed, action, is_first, debug=debug)

        self.dynamics._debug_mode = False

        post = {k: v.detach() for k, v in post.items()}

        if debug:
            print("\n[LOAD DEBUG - POST STATE SUMMARY]")
            for k, v in post.items():
                print(k, tuple(v.shape))
            max_t = min(5, T)
            for t in range(max_t):
                deter_norm = float(post["deter"][0, t].norm().item())
                stoch_norm = float(post["stoch"][0, t].float().norm().item())
                print(
                    f"post t={t:02d} "
                    f"deter_norm={deter_norm:.4f} "
                    f"stoch_norm={stoch_norm:.4f}"
                )

        return post, embed, data

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        if self._config.cont_type == 'cont':
            obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1) # continue predictor
        elif self._config.cont_type == 'mort2':
            cont = torch.where((obs['is_terminal'] == 1) & (obs['mortality'] == 1), torch.tensor(0.0, device=self._config.device), torch.tensor(1.0, device=self._config.device))
            obs["cont"] = cont.unsqueeze(-1) # mortality predictor
        elif self._config.cont_type == 'mort3':
            status = torch.full_like(obs['is_terminal'], 2, dtype=torch.long)
            status[(obs['is_terminal'] == 1) & (obs['mortality'] == 1)] = 0
            status[(obs['is_terminal'] == 1) & (obs['mortality'] == 0)] = 1
            obs['cont'] = F.one_hot(status, num_classes=3) # mortality/discharged/in ICU

        reward = obs[self._config.reward_key].clone()

        if getattr(self._config, "terminal_reward_only", False):
            reward[obs["is_terminal"] == 0] = 0.0

        if getattr(self._config, "terminal_reward_to_pm1", False):
            terminal = obs["is_terminal"].bool()
            death = terminal & (obs["mortality"] == 1)
            survival = terminal & (obs["mortality"] == 0)

            reward[death] = -1.0
            reward[survival] = 1.0

        obs["reward"] = reward.unsqueeze(-1)

        return obs

class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
        feat=None,
        data=None,
        use_history=False,
    ):
        self._update_slow_target()
        metrics = {}
        self._valuenorm = tools.ValueNorm()

        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        feat, action, reward_real_full, cont_real_full = map(
            swap, (feat, data["action"], data["reward"], data["cont"])
        )

        # -------------------------------------------------
        # CASE 1: Hybrid phase-1 training (paper-consistent)
        # -------------------------------------------------
        if use_history:
            # Dataset semantics:
            # feat[t]   = s_t
            # action[t] = action that led to s_t
            #
            # Therefore the real aligned tuples are:
            # current state  : s_t      -> feat[:-1]
            # next action    : a_{t+1}  -> action[1:]
            # next reward    : r_{t+1}  -> reward[1:]
            # next cont      : c_{t+1}  -> cont[1:]

            feat_real = feat[:-1]                # (T-1, B, D) current real states s_t
            n_real = feat_real.shape[0]
            action_real = action[1:]             # (T-1, B, A) next real actions a_{t+1}
            reward_real = reward_real_full[1:]   # (T-1, B, 1) next real rewards r_{t+1}
            if self._config.cont_type == "mort3":
                cont_real = cont_real_full[1:][..., 2:3]
            else:
                cont_real = cont_real_full[1:]       # (T-1, B, 1) or (T-1, B, 3)

            # Start imagination from the final real latent state s_T
            init = {k: v[:, -1] for k, v in start.items()}

            imag_feat, imag_state, imag_action = self._imagine_in_time(
                init,
                self.actor,
                self._config.imag_time,
            )
            # imag_feat   : (tau, B, D) current imagined states
            # imag_state  : dict of next imagined states after each imagined action
            # imag_action : (tau, B, A)

            imag_next_feat = self._world_model.dynamics.get_feat(imag_state)

            # Predicted rewards for imagined transitions
            reward_imag = self._world_model.heads["reward"](imag_next_feat).mode()

            # Predicted continuation for imagined transitions
            if self._config.cont_type == "mort3":
                cont_imag = self._world_model.heads["cont"](imag_next_feat).probs[..., 2, None]
            else:
                cont_imag = self._world_model.heads["cont"](imag_next_feat).mean

            # Current-state sequence for actor/critic
            # [real current states] + [imagined current states]
            feat_hybrid = torch.cat([feat_real, imag_feat], dim=0)

            # Next actions aligned with feat_hybrid
            action_hybrid = torch.cat([action_real, imag_action], dim=0)

            # Rewards aligned with feat_hybrid transitions
            reward_hybrid = torch.cat([reward_real, reward_imag], dim=0)

            # Continuations aligned with feat_hybrid transitions
            cont_hybrid = torch.cat([cont_real, cont_imag], dim=0)

            # Convert continuation to scalar discount
            if self._config.cont_type == "mort3":
                discount_hybrid = self._config.discount * cont_hybrid[..., 2, None] \
                    if cont_hybrid.shape[-1] == 3 else self._config.discount * cont_hybrid
            else:
                discount_hybrid = self._config.discount * cont_hybrid

            with tools.RequiresGrad(self.actor):
                policy = self.actor(feat_hybrid.detach())
                actor_ent = policy.entropy()

                target, weights, base = self._compute_target_hybrid(
                    feat_hybrid,
                    reward_hybrid,
                    discount_hybrid,
                    bootstrap_feat=imag_next_feat[-1].detach(),
                )

                actor_loss, mets = self._compute_actor_loss_hybrid(
                    feat_hybrid,
                    action_hybrid,
                    target,
                    weights,
                    base,
                    n_real=n_real,
                )

                entropy_loss = -self._config.actor["entropy"] * actor_ent.mean()
                actor_loss = actor_loss + entropy_loss
                metrics.update(mets)

            with tools.RequiresGrad(self.value):
                value_dist = self.value(feat_hybrid.detach())
                target_tensor = torch.stack(target, dim=1)

                value_loss = -value_dist.log_prob(target_tensor.detach())

                if self._config.critic["slow_target"]:
                    slow_target = self._slow_value(feat_hybrid.detach()).mode().detach()
                    value_loss -= value_dist.log_prob(slow_target)

                value_loss = torch.mean(weights * value_loss[..., None] if value_loss.ndim == 2 else weights * value_loss)

            metrics.update(tools.tensorstats(value_dist.mode(), "value"))
            metrics.update(tools.tensorstats(target_tensor, "target"))
            metrics.update(tools.tensorstats(reward_hybrid, "hybrid_reward"))
            metrics["actor_entropy"] = to_np(actor_ent.mean())

            with tools.RequiresGrad(self):
                metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
                metrics.update(self._value_opt(value_loss, self.value.parameters()))

            # per compatibilità con il resto del codice
            return feat_hybrid, imag_state, action_hybrid, weights, metrics

        # -----------------------------------------
        # CASE 2: Pure imagination (Dreamer phase-2)
        # -----------------------------------------
        imag_feat, imag_state, imag_action = self._imagine_in_horizon(
            start,
            self.actor,
            self._config.imag_horizon,
        )

        reward = objective(imag_feat, imag_state, imag_action)

        with tools.RequiresGrad(self.actor):
            policy = self.actor(imag_feat)
            actor_ent = policy.entropy()

            target, weights, base = self._compute_target(
                imag_feat,
                imag_state,
                reward,
            )

            actor_loss, mets = self._compute_actor_loss(
                imag_feat,
                imag_action,
                target,
                weights,
                base,
            )

            actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
            actor_loss = actor_loss.mean()
            metrics.update(mets)

        with tools.RequiresGrad(self.value):
            value = self.value(imag_feat[:-1].detach())
            target = torch.stack(target, dim=1)
            value_loss = -value.log_prob(target.detach())

            if self._config.critic["slow_target"]:
                slow_target = self._slow_value(imag_feat[:-1].detach())
                value_loss -= value.log_prob(slow_target.mode().detach())

            value_loss = value_loss.mean()

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        metrics["actor_entropy"] = to_np(actor_ent.mean())

        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))

        return imag_feat, imag_state, imag_action, weights, metrics
    
    def _train_p1(
        self,
        start,
        feat=None,
        data=None,
    ):
        self._update_slow_target()
        metrics = {}
        self._valuenorm = tools.ValueNorm()

        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        feat, action, reward, cont, delta = map(
            swap, (feat, data["action"], data["reward"], data["cont"], data["delta"])
        )

        # feat[t]   = state s_t
        # action[t] = action that brings to s_t
        feat_prev = feat[:-1]       # s_t
        action_next = action[1:]    # a_t
        reward_next = reward[1:]    # r_{t+1}
        if self._config.cont_type == "mort3":
            cont_next = cont[1:][..., 2:3]   # ICU continuation prob
        else:
            cont_next = cont[1:]       # cont_{t+1}
        delta_next = delta[1:]      # delta_{t+1}
        feat_next = feat[1:]        # s_{t+1}

        with tools.RequiresGrad(self.actor):
            actor_ent = self.actor(feat_prev).entropy()

            target, weights, base = self._compute_target_from_real(
                feat_prev, feat_next, reward_next, cont_next
            )

            actor_loss, mets = self._compute_actor_loss_real(
                feat_prev,
                action_next,
                target,
                weights,
                base,
            )
            actor_loss -= self._config.actor["entropy"] * actor_ent[..., None]
            actor_loss = torch.mean(actor_loss)
            metrics.update(mets)

        with tools.RequiresGrad(self.value):
            repfeat = feat_prev.detach()       # s_t
            nextfeat = feat_next.detach()      # s_{t+1}
            pcont = torch.exp(-delta_next / 5).mean(dim=-1, keepdim=True)

            value = self.value(repfeat).mode()                # V(s_t)
            slowval_next = self._slow_value(nextfeat).mode()  # V_target(s_{t+1})
            boot = slowval_next[-1].detach()

            ret = tools.lambda_return(
                reward=reward_next,                 # r_{t+1}
                value=slowval_next.detach(),        # V(s_{t+1})
                pcont=pcont,                        # cont_{t+1}
                bootstrap=boot,
                lambda_=self._config.discount_lambda,
                axis=0,
            )

            ret = torch.stack(ret, dim=1)
            offset, scale = self._valuenorm.update(ret)
            ret_normed = (ret - offset) / scale

            repval_loss = (value - ret_normed.detach()) ** 2
            repval_loss += self._config.critic["slow_reg"] * (
                self._slow_value(repfeat).mode() - ret_normed.detach()
            ) ** 2

            value_loss = torch.mean(weights * repval_loss)
            metrics["reploss/repval_loss"] = to_np(repval_loss.mean())

        metrics.update(tools.tensorstats(value, "value"))
        metrics.update(tools.tensorstats(ret_normed, "target"))
        metrics.update(tools.tensorstats(reward_next, "imag_reward"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))

        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))

        return weights, metrics

    def _train_p1_td(self, feat, data):
        self._update_slow_target()
        metrics = {}

        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        feat, action, reward, cont = map(
            swap, (feat, data["action"], data["reward"], data["cont"])
        )

        # Dataset semantics:
        # feat[t]   = latent state s_t
        # action[t] = action that led to s_t
        #
        # So for policy learning we need:
        #   input state  = s_t      -> feat[:-1]
        #   target action = a_{t+1} -> action[1:]
        #   reward target = r_{t+1} -> reward[1:]
        #   continuation  = c_{t+1} -> cont[1:]

        feat_prev = feat[:-1]      # s_t
        feat_next = feat[1:]       # s_{t+1}
        action_next = action[1:]   # a_{t+1}
        reward_next = reward[1:]   # r_{t+1}
        if self._config.cont_type == "mort3":
            cont_next = cont[1:][..., 2:3]   # ICU continuation prob
        else:
            cont_next = cont[1:]       # cont_{t+1}

        gamma = self._config.discount

        with tools.RequiresGrad(self.value):
            value_t = self.value(feat_prev)                    # V(s_t)
            value_tp1 = self._slow_value(feat_next).mode().detach()  # target V(s_{t+1})

            target = reward_next + gamma * cont_next * value_tp1
            value_loss = -value_t.log_prob(target.detach())
            value_loss = value_loss.mean()

            metrics["td/value_loss"] = to_np(value_loss)
            metrics["td/target_mean"] = to_np(target.mean())

        with tools.RequiresGrad(self.actor):
            policy = self.actor(feat_prev.detach())            # pi(. | s_t)
            actor_ent = policy.entropy()
            logp = policy.log_prob(action_next)                # log pi(a_{t+1} | s_t)

            baseline = value_t.mode().detach()
            advantage = target.detach() - baseline

            actor_loss = -(logp.unsqueeze(-1) * advantage)
            actor_loss -= self._config.actor["entropy"] * actor_ent.unsqueeze(-1)
            actor_loss = actor_loss.mean()

        metrics.update(tools.tensorstats(value_t.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward_next, "imag_reward"))
        metrics["actor_entropy"] = to_np(actor_ent.mean())

        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))

        return metrics

    def _imagine_in_horizon(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions
    
    def _imagine_in_time(self, start, policy, time):
        dynamics = self._world_model.dynamics

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(time)], (start, None, None)
        )

        return feats, succ, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            if self._config.cont_type == "mort3":
                discount = self._config.discount * self._world_model.heads["cont"](inp).probs[..., 2, None]
            else:
                discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])
        else:
            adv = target - base

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics
    
    def _compute_target_hybrid(self, feat_hybrid, reward_hybrid, discount_hybrid, bootstrap_feat):
        """
        feat_hybrid[t]      = current state s_t
        reward_hybrid[t]    = reward for transition from s_t
        discount_hybrid[t]  = continuation/discount for transition from s_t
        bootstrap_feat      = state after the last transition
        """
        value = self.value(feat_hybrid).mode()
        bootstrap = self._slow_value(bootstrap_feat).mode().detach()

        target = tools.lambda_return(
            reward=reward_hybrid,
            value=value,
            pcont=discount_hybrid,
            bootstrap=bootstrap,
            lambda_=self._config.discount_lambda,
            axis=0,
        )

        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount_hybrid[:1]), discount_hybrid[:-1]], dim=0),
            dim=0,
        ).detach()

        return target, weights, value


    def _compute_actor_loss_hybrid(
        self,
        feat_hybrid,
        action_hybrid,
        target,
        weights,
        base,
        n_real=None,
    ):
        metrics = {}
        policy = self.actor(feat_hybrid.detach())

        target = torch.stack(target, dim=1)

        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base

            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])
        else:
            adv = target - base

        if self._config.imag_gradient == "dynamics":
            actor_target = adv

        elif self._config.imag_gradient == "reinforce":
            actor_target = policy.log_prob(action_hybrid)[..., None] * adv.detach()

        elif self._config.imag_gradient == "both":
            reinforce_term = policy.log_prob(action_hybrid)[..., None] * adv.detach()
            mix = self._config.imag_gradient_mix
            actor_target = mix * adv + (1 - mix) * reinforce_term
            metrics["imag_gradient_mix"] = mix

        else:
            raise NotImplementedError(self._config.imag_gradient)

        actor_loss = -weights * actor_target

        if n_real is not None:
            real_loss = actor_loss[:n_real].mean()
            imag_loss = actor_loss[n_real:].mean()

            real_weight = getattr(self._config, "real_loss_weight", 1.0)
            imag_weight = getattr(self._config, "imag_loss_weight", 0.2)

            actor_loss = real_weight * real_loss + imag_weight * imag_loss

            metrics["actor_loss_real"] = to_np(real_loss)
            metrics["actor_loss_imag"] = to_np(imag_loss)
            metrics["real_loss_weight"] = real_weight
            metrics["imag_loss_weight"] = imag_weight
        else:
            actor_loss = actor_loss.mean()

        logp = policy.log_prob(action_hybrid)
        probs = policy.probs
        argmax_actions = probs.argmax(dim=-1)

        metrics["adv_mean"] = to_np(adv.mean())
        metrics["adv_std"] = to_np(adv.std())
        metrics["adv_min"] = to_np(adv.min())
        metrics["adv_max"] = to_np(adv.max())
        metrics["adv_pos_frac"] = to_np((adv > 0).float().mean())
        metrics["adv_neg_frac"] = to_np((adv < 0).float().mean())
        metrics["adv_abs_mean"] = to_np(adv.abs().mean())

        metrics["target_mean"] = to_np(target.mean())
        metrics["base_mean"] = to_np(base.mean())
        metrics["target_minus_base_mean"] = to_np((target - base).mean())

        metrics["logp_mean"] = to_np(logp.mean())
        metrics["logp_std"] = to_np(logp.std())

        metrics["pi_max_mean"] = to_np(probs.max(dim=-1).values.mean())
        metrics["pi_min_mean"] = to_np(probs.min(dim=-1).values.mean())
        metrics["pi_entropy_mean"] = to_np(policy.entropy().mean())

        counts = torch.bincount(
            argmax_actions.reshape(-1),
            minlength=self._config.num_actions,
        ).float()
        metrics["pi_argmax_frac"] = to_np(counts.max() / counts.sum())
        metrics["pi_argmax_action"] = to_np(torch.argmax(counts))

        if n_real is not None:
            adv_real = adv[:n_real]
            adv_imag = adv[n_real:]

            logp_real = logp[:n_real]
            logp_imag = logp[n_real:]

            probs_real = probs[:n_real]
            probs_imag = probs[n_real:]

            metrics["adv_real_mean"] = to_np(adv_real.mean())
            metrics["adv_real_pos_frac"] = to_np((adv_real > 0).float().mean())
            metrics["adv_imag_mean"] = to_np(adv_imag.mean())
            metrics["adv_imag_pos_frac"] = to_np((adv_imag > 0).float().mean())

            metrics["logp_real_mean"] = to_np(logp_real.mean())
            metrics["logp_imag_mean"] = to_np(logp_imag.mean())

            metrics["pi_max_real_mean"] = to_np(probs_real.max(dim=-1).values.mean())
            metrics["pi_max_imag_mean"] = to_np(probs_imag.max(dim=-1).values.mean())

            metrics["entropy_real_mean"] = to_np(policy.entropy()[:n_real].mean())
            metrics["entropy_imag_mean"] = to_np(policy.entropy()[n_real:].mean())

        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1


class BehaviorPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

        if config.dyn_discrete:
            input_dim = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            input_dim = config.dyn_stoch + config.dyn_deter

        bp = config.behavior_model
        self.policy_type = bp.get("type", "lstm")

        hidden_size = bp.get("hidden_size", 16)
        num_layers = bp.get("num_layers", 1)

        if self.policy_type == "lstm":
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, config.num_actions)

        elif self.policy_type == "mlp":
            layers = []
            in_dim = input_dim

            for _ in range(num_layers):
                layers.append(nn.Linear(in_dim, hidden_size))
                layers.append(nn.ReLU())
                in_dim = hidden_size

            layers.append(nn.Linear(hidden_size, config.num_actions))
            self.net = nn.Sequential(*layers)

        else:
            raise ValueError(f"Unknown behavior_model.type: {self.policy_type}")

        self._opt = tools.Optimizer(
            "behavior",
            self.parameters(),
            lr=bp.get("lr", 1e-4),
            eps=bp.get("eps", 1e-8),
            clip=bp.get("grad_clip", 100.0),
            wd=config.weight_decay,
            opt=config.opt,
        )

    def _init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0, c0

    def forward(self, feat, is_first=None):
        """
        feat: [B, T, D]
        returns distribution over actions for each timestep.
        """

        if self.policy_type == "mlp":
            B, T, D = feat.shape
            x = feat.reshape(B * T, D)
            logits = self.net(x)
            logits = logits.reshape(B, T, -1)
            return tools.OneHotDist(logits)

        if self.policy_type == "lstm":
            B, T, _ = feat.shape
            device = feat.device

            if is_first is None:
                h, _ = self.lstm(feat)
                logits = self.head(h)
                return tools.OneHotDist(logits)

            hx, cx = self._init_hidden(B, device)
            outputs = []

            for t in range(T):
                reset_mask = is_first[:, t].bool()

                if reset_mask.any():
                    hx = hx.clone()
                    cx = cx.clone()
                    hx[:, reset_mask] = 0.0
                    cx[:, reset_mask] = 0.0

                out_t, (hx, cx) = self.lstm(feat[:, t:t + 1, :], (hx, cx))
                outputs.append(out_t)

            h = torch.cat(outputs, dim=1)
            logits = self.head(h)
            return tools.OneHotDist(logits)

        raise ValueError(f"Unknown behavior_model.type: {self.policy_type}")

    def train_batch(self, feat, action_onehot, is_first):
        self.train()

        # Learn pi_b(a_{t+1} | s_t)
        feat_input = feat[:, :-1]
        action_target = action_onehot[:, 1:]
        is_first_input = is_first[:, :-1]

        with tools.RequiresGrad(self):
            if self.policy_type == "lstm":
                dist = self(feat_input, is_first_input)
            else:
                dist = self(feat_input)

            loss = -dist.log_prob(action_target).mean()
            metrics = self._opt(loss, self.parameters(), retain_graph=False)

        with torch.no_grad():
            pred = dist.mode().argmax(-1)
            true = action_target.argmax(-1)
            accuracy = (pred == true).float().mean()
            p_clin = torch.exp(dist.log_prob(action_target)).mean()
            entropy = dist.entropy().mean()

        metrics["behavior_loss"] = to_np(loss)
        metrics["behavior_acc"] = to_np(accuracy)
        metrics["behavior_avg_p"] = to_np(p_clin)
        metrics["behavior_entropy"] = to_np(entropy)        

        return metrics