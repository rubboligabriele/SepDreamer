import time
import os
import json
import pathlib
import random
import pickle

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd
from torch.utils.tensorboard import SummaryWriter

to_np = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)

def bt_flatten(x):
    """Flatten (B, T, ...) -> (B*T, ...)"""
    return x.reshape([-1] + list(x.shape[2:]))

def bt_unflatten(x, B, T):
    """Unflatten (B*T, ...) -> (B, T, ...)"""
    return x.reshape([B, T] + list(x.shape[1:]))

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)

class Logger:
    def __init__(self, logdir):
        self._logdir = logdir
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = value

    def write(self, step):
        scalars = list(self._scalars.items())
        # print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            if "/" not in name:
                self._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._writer.add_scalar(name, value, step)
        for name, value in self._images.items():
            self._writer.add_figure(name, value, step)

        self._writer.flush()
        self._scalars = {}
        self._images = {}

def extract_best_from_json(logdir):
    to_minimize = ["recon_error", "reward_error", "ai_mortality"]

    to_maximize = [
        "wis",
        "wpdis",
        "cwpdis",
        "ess",
        "mortality_decrease",
    ]

    best = {k: float("inf") for k in to_minimize}
    best.update({k: float("-inf") for k in to_maximize})
    best_steps = {}

    with open(logdir / "metrics.jsonl", "r") as f:
        for line in f:
            record = json.loads(line)
            step = record.get("step", None)
            for k in to_minimize:
                if k in record and record[k] < best[k]:
                    best[k] = record[k]
                    best_steps[k] = step
            for k in to_maximize:
                if k in record and record[k] > best[k]:
                    best[k] = record[k]
                    best_steps[k] = step

    summary = {}
    for k in best:
        summary[k] = best[k]
        summary[f"{k}_step"] = best_steps.get(k, None)

    return summary

def from_generator(generator, batch_size):
    while True:
        batch = [next(generator) for _ in range(batch_size)]

        data = {}
        keys = batch[0].keys()

        for key in keys:
            arrays = [x[key] for x in batch]
            data[key] = np.stack(arrays, axis=0)

        yield data

def sample_episodes(episodes, length, seed=0):
    np_random = np.random.RandomState(seed)

    # iterate over real episodes data (keys = stayids)
    episode_list = list(episodes.values())
    if len(episode_list) == 0:
        raise ValueError("No episodes available.")

    time_keys = [
        "timestep",
        "features",
        "action",
        "reward",
        "mask",
        "delta",
        "is_first",
        "is_terminal",
        "discount",
        "mortality",
    ]

    # array of episode lengths, used to sample episodes with probability proportional to their length
    episode_lengths = np.array(
        [len(ep["timestep"]) for ep in episode_list],
        dtype=np.int64,
    )

    # conversion fo float64 because division with int64 can give 0, butr we need a probability distribution (ex. 0.35, 0.65)
    p = episode_lengths.astype(np.float64)
    p = p / p.sum()

    while True:
        size = 0
        chunks = []
        first_flags = []

        # lenght is the number of timesteps we want to sample from the episodes, we keep sampling until we reach this length (usually 50)
        while size < length:
            ep_idx = np_random.choice(len(episode_list), p=p)
            episode = episode_list[ep_idx]
            total = episode_lengths[ep_idx]

            # if the episode is too short, we skip it and sample another one
            if total < 2:
                continue

            if size == 0:
                # if we are at the beginning of the sampling, we can sample a random index in the episode and take a chunk of length "length" from there
                index = int(np_random.randint(0, total - 1))
                end = min(index + length, total)
            else:
                # to complete the batch we sample other episodes, but always starting from step 0...because of "is_first".
                # We need the model to understand it's not a single episode.
                index = 0
                possible = length - size
                end = min(index + possible, total)

            chunk = {
                k: episode[k][index:end].copy()
                for k in time_keys
            }

            chunks.append(chunk)
            first_flags.append(size)
            size += len(chunk["timestep"])

        # reset is_first flags for the concatenated chunks, only the first timestep of the first chunk is marked as True, all others are False
        ret = {}
        for k in time_keys:
            ret[k] = np.concatenate([c[k] for c in chunks], axis=0)

        ret["is_first"][:] = False
        for pos in first_flags:
            ret["is_first"][pos] = True

        yield ret

def sample_episodes_single(episodes, length, seed=0):
    """Like sample_episodes but never concatenates multiple episodes.
    Episodes shorter than length are skipped. For the policy p1, this
    prevents lambda-return targets from crossing episode boundaries."""
    np_random = np.random.RandomState(seed)

    episode_list = list(episodes.values())
    if len(episode_list) == 0:
        raise ValueError("No episodes available.")

    time_keys = [
        "timestep",
        "features",
        "action",
        "reward",
        "mask",
        "delta",
        "is_first",
        "is_terminal",
        "discount",
        "mortality",
    ]

    eligible = [(ep, len(ep["timestep"])) for ep in episode_list if len(ep["timestep"]) >= length]
    if len(eligible) == 0:
        raise ValueError(f"No episodes with length >= {length}.")

    ep_lengths = np.array([l for _, l in eligible], dtype=np.float64)
    p = ep_lengths / ep_lengths.sum()

    while True:
        idx = np_random.choice(len(eligible), p=p)
        episode, total = eligible[idx]
        index = int(np_random.randint(0, total - length + 1))
        chunk = {k: episode[k][index:index + length].copy() for k in time_keys}
        chunk["is_first"][:] = False
        chunk["is_first"][0] = True
        yield chunk


class EpisodeSampler:
    def __init__(self, episodes, sample_size=100):
        self.episodes = episodes
        self.sample_size = sample_size

    def __iter__(self):
        keys = random.sample(list(self.episodes.keys()), self.sample_size)
        sampled = [(k, self.episodes[k]) for k in keys]
        return iter(sampled)

def load_episode_npz(filepath):
    data = np.load(filepath)
    return {key: data[key] for key in data.files}

def load_all_episode_keys(saved_eps_dir):
    """
    Takes the icustayid of every .npz episode in the episodes directory, using the filename without the .npz extension as the icustayid
    """
    filenames = [f for f in os.listdir(saved_eps_dir) if f.endswith('.npz')]
    stay_ids = [fname[:-4] for fname in filenames]
    return stay_ids

def load_split_episodes(saved_eps_dir, stay_ids, cache_path=None):
    """
    Loads episodes for the given stay_ids from the saved_eps_dir.
    If cache_path is provided but the cached files don't exist yet, it will save the loaded episodes to a pickle file at cache_path for faster loading next time.
    If the pickle file already exists, it will load the episodes from the pickle file instead of loading from the .npz files, for higher efficiency.
    """
    if cache_path is not None and os.path.exists(cache_path):
        print(f"[DataLoader] Loading cached episodes from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            episodes = pickle.load(f)
        return episodes

    # if cache is not available, load episodes from .npz files and save to cache if cache_path is provided
    episodes = {}
    total = len(stay_ids)

    print(f"[DataLoader] Loading {total} episodes from {saved_eps_dir}", flush=True)

    for i, stay_id in enumerate(stay_ids, 1):
        # recreate files name using stayids
        filepath = os.path.join(saved_eps_dir, f"{stay_id}.npz")
        episode = load_episode_npz(filepath)

        # create a dictionary with stay_id as key and REAL episode as value
        episodes[stay_id] = episode

        # for logging progress, print every 100 episodes or at the first episode
        if i % 100 == 0 or i == 1:
            print(f"[DataLoader] Loaded {i}/{total}", flush=True)

    if cache_path is not None:
        print(f"[DataLoader] Saving cache to {cache_path}", flush=True)
        with open(cache_path, "wb") as f:
            pickle.dump(episodes, f)

    return episodes

class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape).detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class DiscDist:
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255, device=logits.device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        return self.mean()

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)

def _aggregate(distance, agg):
    dims = list(range(len(distance.shape)))[2:]
    if agg == "mean":
        return distance.mean(dims)
    elif agg == "sum":
        return distance.sum(dims)
    elif agg == "raw":
        return distance
    raise NotImplementedError(agg)

class MSEDist:
    def __init__(self, mode, agg="raw"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        return -_aggregate(distance, self._agg)


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="raw", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mean(self):
        return symexp(self._mode)

    def mode(self):
        return self.mean()

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        return -_aggregate(distance, self._agg)

class ContDist:
    def __init__(self, dist=None, absmax=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)
        return torch.sum(log_probs0 * (1 - x) + log_probs1 * x, -1)


class UnnormalizedHuber(torchd.normal.Normal):
    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(
            torch.sqrt((event - self.mean) ** 2 + self._threshold**2) - self._threshold
        )

    def mode(self):
        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torchd.Transform):
    def __init__(self, validate_args=False, name="tanh"):
        super().__init__()

    def _forward(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = torch.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        # (inputs, pcont) -> (inputs[index], pcont[index])
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.flip(outputs, [1])
    outputs = torch.unbind(outputs, dim=0)
    return outputs

class ValueNorm:
    def __init__(self, momentum=0.99, eps=1e-8):
        self.momentum = momentum
        self.eps = eps
        self.mean = None
        self.std = None
        self.first = True

    def update(self, values):
        # values: (T, B, 1) or (B, 1) or (N,)
        mean = values.mean().detach()
        std = values.std(unbiased=False).detach()

        if self.first:
            self.mean = mean
            self.std = std
            self.first = False
        else:
            self.mean = self.momentum * self.mean + (1 - self.momentum) * mean
            self.std = self.momentum * self.std + (1 - self.momentum) * std

        return self.stats()

    def stats(self):
        return self.mean, self.std.clamp(min=self.eps)

def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        wd_pattern=r".*",
        opt="adam",
        use_amp=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params, retain_graph=True):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._opt.zero_grad()
        self._scaler.scale(loss).backward(retain_graph=retain_graph)
        self._scaler.unscale_(self._opt)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)

        if self._wd:
            self._apply_weight_decay(params)

        self._scaler.step(self._opt)
        self._scaler.update()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = to_np(norm)
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])

    # Handle empty sequence
    if inputs[0].shape[0] == 0:
        if isinstance(start, dict):
            return [{k: v.new_empty((0,) + v.shape) for k, v in start.items()}]
        elif isinstance(start, (list, tuple)):
            outputs = []
            for item in start:
                if isinstance(item, dict):
                    outputs.append({k: v.new_empty((0,) + v.shape) for k, v in item.items()})
                else:
                    outputs.append(item.new_empty((0,) + item.shape))
            return outputs
        else:
            return [start.new_empty((0,) + start.shape)]

    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if isinstance(last, dict):
                outputs = {
                    key: value.clone().unsqueeze(0) for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if isinstance(_last, dict):
                        outputs.append(
                            {
                                key: value.clone().unsqueeze(0)
                                for key, value in _last.items()
                            }
                        )
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if isinstance(last, dict):
                for key in last.keys():
                    outputs[key] = torch.cat(
                        [outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if isinstance(last[j], dict):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat(
                                [outputs[j][key], last[j][key].unsqueeze(0)], dim=0
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)], dim=0
                        )

    if isinstance(last, dict):
        outputs = [outputs]
    return outputs


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until

def _compute_fan(m):
    if isinstance(m, nn.Linear):
        return m.in_features, m.out_features
    space = m.kernel_size[0] * m.kernel_size[1]
    return space * m.in_channels, space * m.out_channels

def weight_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        in_num, out_num = _compute_fan(m)
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            in_num, out_num = _compute_fan(m)
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
    return f


def tensorstats(tensor, prefix=None):
    metrics = {
        "mean": to_np(torch.mean(tensor)),
        "std": to_np(torch.std(tensor)),
        "min": to_np(torch.min(tensor)),
        "max": to_np(torch.max(tensor)),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(
    obj, path="", optimizers_state_dicts=None, visited=None
):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    else:
        visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update(
            {k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr}
        )
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(
                    attr, new_path, optimizers_state_dicts, visited
                )
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)

def save_model(agent, model_name, dir, epoch):
    if model_name == "all":
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, pathlib.Path(dir).expanduser() / f"agent_{epoch}.pt")

    else:
        assert hasattr(agent, f"_{model_name}"), f"Agent has no attribute '{model_name}'"
        model = getattr(agent, f"_{model_name}")

        save_dict = {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "optim_state_dicts": recursively_collect_optim_state_dict(model)
        }

        torch.save(save_dict, pathlib.Path(dir).expanduser() / f"{model_name}_{epoch}.pt")

def load_model(agent, model_name, dir, epoch, device, actor_lr=None, value_lr=None):
    if model_name == "all":
        items_to_load = torch.load(
            pathlib.Path(dir).expanduser() / f"agent_{epoch}.pt",
            map_location=device,
            weights_only=False
        )
        agent.load_state_dict(items_to_load["agent_state_dict"], strict=False)
        if actor_lr:
            items_to_load["optims_state_dict"]['_task_behavior._actor_opt._opt']['param_groups'][0]['lr'] = actor_lr
            items_to_load["optims_state_dict"]['_task_behavior._value_opt._opt']['param_groups'][0]['lr'] = value_lr
        recursively_load_optim_state_dict(agent, items_to_load["optims_state_dict"])
    else:
        checkpoint = torch.load(
            pathlib.Path(dir).expanduser() / f"{model_name}_{epoch}.pt",
            map_location=device,
            weights_only=False,
        )
        assert checkpoint["model_name"] == model_name, \
            f"Expected {model_name}, but file contains {checkpoint['model_name']}"

        model = getattr(agent, f"_{model_name}")
        model.load_state_dict(checkpoint["state_dict"])
        recursively_load_optim_state_dict(model, checkpoint["optim_state_dicts"])

def sliding_mean(x, window=5):
    return np.array([
        np.mean(x[max(0, i - window + 1):min(len(x), i + window + 1)])
        for i in range(len(x))
    ])

def find_nearest_Q(Q_mean, res_dt):
    idx = np.argmin(np.abs(res_dt['q_value'] - Q_mean))
    return res_dt.iloc[idx]


def plot_mortality_vs_value(value_values, mortality, num_bins=20, xlabel="Estimated Value"):
    # Trim to middle 98% quantile range
    q01, q99 = np.quantile(value_values, [0.01, 0.99])
    mask = (value_values >= q01) & (value_values <= q99)
    value_values = value_values[mask]
    mortality = mortality[mask.squeeze()]

    # Define bin edges and centers
    bin_edges = np.linspace(q01, q99, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Assign each value to a bin
    bin_indices = np.digitize(value_values, bin_edges) - 1

    mortality_means = np.empty(num_bins)
    mortality_stds = np.empty(num_bins)
    mortality_means[:] = np.nan
    mortality_stds[:] = np.nan

    for i in range(num_bins):
        in_bin = bin_indices == i
        if np.sum(in_bin) >= 2:
            mortality_means[i] = np.mean(mortality[in_bin])
            mortality_stds[i] = sem(mortality[in_bin])

    valid = ~np.isnan(mortality_means)
    bin_centers = bin_centers[valid]
    mortality_means = mortality_means[valid]
    mortality_stds = mortality_stds[valid]

    smoothed = sliding_mean(mortality_means)
    smoothed_std = sliding_mean(mortality_stds)

    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor='white')
    ax.plot(bin_centers, smoothed)
    ax.fill_between(
        bin_centers,
        smoothed - smoothed_std,
        smoothed + smoothed_std,
        color='#ADD8E6'
    )
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel("Mortality", fontsize=15)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.tick_params(labelsize=15)
    ax.grid()
    fig.tight_layout()

    return fig, bin_centers, smoothed, smoothed_std


def compute_mortality_curve(value_values, mortality, num_bins=20):
    q01, q99 = np.quantile(value_values, [0.01, 0.99])
    mask = (value_values >= q01) & (value_values <= q99)
    value_values = value_values[mask]
    mortality = mortality[mask.squeeze()]

    bin_edges = np.linspace(q01, q99, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_indices = np.digitize(value_values, bin_edges) - 1

    mortality_means = np.full(num_bins, np.nan)
    for i in range(num_bins):
        in_bin = bin_indices == i
        if np.sum(in_bin) >= 2:
            mortality_means[i] = np.mean(mortality[in_bin])

    valid = ~np.isnan(mortality_means)
    bin_centers = bin_centers[valid]
    smoothed = sliding_mean(mortality_means[valid])

    return bin_centers, smoothed


def plot_critic_vs_feature(feature_values, critic_values, xlabel="Feature", num_bins=15):
    """Bin feature_values and show mean critic value per bin with SEM shading.

    Useful to check whether the critic assigns higher values to healthier states
    (e.g., low SOFA → high V(s), high phys_return → high V(s)).
    """
    feature_values = np.array(feature_values, dtype=np.float64)
    critic_values = np.array(critic_values, dtype=np.float64)

    q01, q99 = np.quantile(feature_values, [0.01, 0.99])
    mask = (feature_values >= q01) & (feature_values <= q99)
    feature_values = feature_values[mask]
    critic_values = critic_values[mask]

    bin_edges = np.linspace(q01, q99, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_indices = np.digitize(feature_values, bin_edges) - 1

    value_means = np.full(num_bins, np.nan)
    value_sems = np.full(num_bins, np.nan)

    for i in range(num_bins):
        in_bin = bin_indices == i
        if np.sum(in_bin) >= 2:
            value_means[i] = np.mean(critic_values[in_bin])
            value_sems[i] = sem(critic_values[in_bin])

    valid = ~np.isnan(value_means)
    bin_centers = bin_centers[valid]
    value_means = value_means[valid]
    value_sems = value_sems[valid]

    smoothed = sliding_mean(value_means)
    smoothed_sem = sliding_mean(value_sems)

    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor='white')
    ax.plot(bin_centers, smoothed)
    ax.fill_between(
        bin_centers,
        smoothed - smoothed_sem,
        smoothed + smoothed_sem,
        color='#ADD8E6',
    )
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel("Critic Value V(s)", fontsize=15)
    ax.tick_params(labelsize=15)
    ax.grid()
    fig.tight_layout()
    plt.show()

    return fig


def plot_critic_vs_actual_return(critic_values, actual_returns, discount):
    """Scatter V(s) vs actual discounted return from the same starting state.

    The critic is well-calibrated when points cluster around the diagonal.
    Returns fig and the Pearson correlation coefficient.
    """
    from scipy.stats import pearsonr
    critic_values = np.array(critic_values, dtype=np.float64)
    actual_returns = np.array(actual_returns, dtype=np.float64)

    mask = np.isfinite(critic_values) & np.isfinite(actual_returns)
    cv = critic_values[mask]
    ar = actual_returns[mask]

    corr, _ = pearsonr(cv, ar) if len(cv) > 2 else (float("nan"), None)

    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor="white")
    ax.scatter(ar, cv, alpha=0.3, s=8, color="#4C72B0")

    lo = min(ar.min(), cv.min())
    hi = max(ar.max(), cv.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="diagonal")

    ax.set_xlabel(f"Lambda-return target R^λ(s₄)", fontsize=13)
    ax.set_ylabel("Critic V(s₄)", fontsize=13)
    ax.set_title(f"Pearson r = {corr:.3f}", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.4)
    ax.legend(fontsize=11)
    fig.tight_layout()
    plt.show()

    return fig, corr


def plot_recon_error_per_feature(
    feat_mse_post, feat_mse_prior, feat_names,
    feat_mse_post_death=None, feat_mse_prior_death=None,
    feat_mse_post_surv=None, feat_mse_prior_surv=None,
):
    """
    Bar chart of per-feature MSE for post (observed) and prior (imagined) rollout.
    Optionally overlays death vs survival breakdown.
    Returns (fig_post, fig_prior, fig_delta).
    """
    n = len(feat_names)
    idx = np.arange(n)

    def _bar_fig(mse_main, label_main, mse_death=None, mse_surv=None, title=""):
        order = np.argsort(-mse_main)
        fig, ax = plt.subplots(figsize=(max(12, n * 0.35), 5), facecolor="white")
        ax.bar(idx, mse_main[order], label=label_main, alpha=0.75)
        if mse_death is not None and mse_surv is not None:
            ax.plot(idx, mse_death[order], "v", color="red",   markersize=5, label="death")
            ax.plot(idx, mse_surv[order],  "^", color="green", markersize=5, label="survival")
        ax.set_xticks(idx)
        ax.set_xticklabels(np.array(feat_names)[order], rotation=90, fontsize=8)
        ax.set_ylabel("MSE")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.4)
        fig.tight_layout()
        return fig

    fig_post  = _bar_fig(feat_mse_post,  "post",  feat_mse_post_death,  feat_mse_post_surv,  "Reconstruction MSE per feature (post — observed steps)")
    fig_prior = _bar_fig(feat_mse_prior, "prior", feat_mse_prior_death, feat_mse_prior_surv, "Reconstruction MSE per feature (prior — imagined rollout)")

    # delta: prior - post, sorted by worst degradation
    delta = feat_mse_prior - feat_mse_post
    order_d = np.argsort(-delta)
    fig_delta, ax = plt.subplots(figsize=(max(12, n * 0.35), 5), facecolor="white")
    colors = ["red" if d > 0 else "steelblue" for d in delta[order_d]]
    ax.bar(idx, delta[order_d], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(np.array(feat_names)[order_d], rotation=90, fontsize=8)
    ax.set_ylabel("MSE prior − post")
    ax.set_title("Rollout degradation per feature (red = worse in imagined rollout)")
    ax.grid(axis="y", alpha=0.4)
    fig_delta.tight_layout()

    return fig_post, fig_prior, fig_delta


def calculate_estimated_mortality(ai_values, bin_centers, smoothed, smoothed_sem):
    # Map each AI patient's return to the calibration curve, then average across patients.
    # This matches the standard literature approach (e.g. Komorowski 2018, Raghu 2017):
    # bin each patient individually rather than interpolating the population mean.
    per_patient_mortality = np.interp(ai_values, bin_centers, smoothed)
    per_patient_std = np.interp(ai_values, bin_centers, smoothed_sem)

    ai_mortality = float(np.mean(per_patient_mortality))
    ai_std = float(np.mean(per_patient_std))

    print("\n[MORTALITY INTERP DEBUG]")
    print("ai_return_mean =", float(np.mean(ai_values)))
    print("ai_return_std =", float(np.std(ai_values)))
    print("curve_return_min =", float(np.min(bin_centers)))
    print("curve_return_max =", float(np.max(bin_centers)))
    print("frac_below_curve =", float(np.mean(ai_values < np.min(bin_centers))))
    print("frac_above_curve =", float(np.mean(ai_values > np.max(bin_centers))))
    print("ai_mortality_estimated =", round(ai_mortality * 100, 2))

    return ai_mortality, ai_std

def compute_accuracy(logits, targets):
    """
    logits: shape [B, T, 1] or [B, T, C]
    targets: shape [B], values in {0, 1} or {0, 1, 2}
    """
    if logits.shape[-1] == 3:
        # Multi-class case: logits shape [B, C]
        preds = torch.argmax(logits, dim=-1)
        targets = torch.argmax(targets, dim=-1)
    else: 
        preds = logits

    correct = (preds == targets).sum().item()
    total = targets.numel()
    
    return round(correct / total * 100, 2)

def compute_ope_trajectory(
    pi_ai_list,
    pi_b_list,
    reward_list,
    gamma,
    prob_eps=1e-6,
    rho_max=5.0,
    max_ope_steps=30,
):
    """
    Build one trajectory for OPE.

    Definitions used here:
      - rho_raw_t   = pi_e(a_t|s_t) / pi_b(a_t|s_t)
      - rho_t       = clipped rho_raw_t
      - cum_rho_t   = prod_{i<=t} rho_i
      - disc_reward_t = gamma^t * r_t

    Notes:
      - All returned aggregate metrics downstream are computed from CLIPPED weights.
      - Raw cumulative weights are also stored for debugging.
    """
    pi_ai = np.asarray(pi_ai_list, dtype=np.float64).reshape(-1)
    pi_b = np.asarray(pi_b_list, dtype=np.float64).reshape(-1)
    rewards = np.asarray(reward_list, dtype=np.float64).reshape(-1)

    if max_ope_steps is not None:
        pi_ai = pi_ai[:max_ope_steps]
        pi_b = pi_b[:max_ope_steps]
        rewards = rewards[:max_ope_steps]

    T = len(rewards)
    if T == 0:
        return None

    # avoid exact zeros
    pi_ai = np.clip(pi_ai, prob_eps, 1.0)
    pi_b = np.clip(pi_b, prob_eps, 1.0)

    rho_raw = pi_ai / pi_b
    rho = np.clip(rho_raw, 0.0, rho_max)

    cum_rho_raw = np.cumprod(rho_raw)
    cum_rho = np.cumprod(rho)

    discounts = gamma ** np.arange(T, dtype=np.float64)
    disc_rewards = discounts * rewards

    traj_return = float(np.sum(disc_rewards))

    # trajectory-wise IS quantities
    is_weight_raw = float(cum_rho_raw[-1])
    is_weight = float(cum_rho[-1])

    # per-decision trajectory contribution
    pdis_raw = float(np.sum(cum_rho_raw * disc_rewards))
    pdis = float(np.sum(cum_rho * disc_rewards))

    return {
        "rho_raw": rho_raw,
        "rho": rho,
        "cum_rho_raw": cum_rho_raw,
        "cum_rho": cum_rho,
        "disc_rewards": disc_rewards,
        "traj_return": traj_return,
        "is_weight_raw": is_weight_raw,
        "is_weight": is_weight,
        "pdis_raw": pdis_raw,
        "pdis": pdis,
        "debug": {
            "T": int(T),
            "reward_sum": float(np.sum(rewards)),
            "disc_reward_sum": float(np.sum(disc_rewards)),
            "pi_ai_min": float(pi_ai.min()),
            "pi_ai_max": float(pi_ai.max()),
            "pi_b_min": float(pi_b.min()),
            "pi_b_max": float(pi_b.max()),
            "rho_raw_min": float(rho_raw.min()),
            "rho_raw_max": float(rho_raw.max()),
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
            "cum_rho_raw_last": float(cum_rho_raw[-1]),
            "cum_rho_last": float(cum_rho[-1]),
            "n_pi_b_lt_1e-2": int(np.sum(pi_b < 1e-2)),
            "n_pi_b_lt_1e-3": int(np.sum(pi_b < 1e-3)),
            "n_pi_b_lt_1e-4": int(np.sum(pi_b < 1e-4)),
            "n_pi_b_lt_1e-6": int(np.sum(pi_b < 1e-6)),
            "n_rho_raw_gt_10": int(np.sum(rho_raw > 10.0)),
            "n_rho_raw_gt_100": int(np.sum(rho_raw > 100.0)),
            "n_rho_raw_gt_1000": int(np.sum(rho_raw > 1000.0)),
        },
    }


def finalize_ope(ope_trajs, debug=False, top_k=5):
    """
    Aggregate OPE metrics across trajectories.

    Returned metrics:
      - is:     ordinary trajectory-wise IS     (clipped)
      - wis:    weighted trajectory-wise IS     (clipped)
      - pdis:   ordinary per-decision IS        (clipped)
      - wpdis:  weighted PDIS averaged over N   (same scale as ordinary mean contribution)
      - cwpdis: self-normalized / cumulative weighted PDIS
      - ess:    ESS from final clipped trajectory weights
      - ess_min / ess_mean: per-step ESS summary from clipped weights
    """
    if len(ope_trajs) == 0:
        return {
            "is": 0.0,
            "wis": 0.0,
            "pdis": 0.0,
            "wpdis": 0.0,
            "cwpdis": 0.0,
            "ess": 0.0,
            "ess_min": 0.0,
            "ess_mean": 0.0,
        }

    n = len(ope_trajs)

    final_weights = np.array(
        [traj["cum_rho"][-1] for traj in ope_trajs],
        dtype=np.float64,
    )
    final_weights_raw = np.array(
        [traj["cum_rho_raw"][-1] for traj in ope_trajs],
        dtype=np.float64,
    )
    traj_returns = np.array(
        [traj["traj_return"] for traj in ope_trajs],
        dtype=np.float64,
    )

    # trajectory-wise IS
    is_est = float(np.mean(final_weights * traj_returns))
    wis = float(np.sum(final_weights * traj_returns) / (np.sum(final_weights) + 1e-8))

    # per-decision aggregation
    max_len = max(len(traj["cum_rho"]) for traj in ope_trajs)

    pdis = 0.0
    wpdis = 0.0
    cwpdis = 0.0
    ess_t_list = []

    for t in range(max_len):
        weights_t = []
        weighted_rewards_t = []

        for traj in ope_trajs:
            if t < len(traj["cum_rho"]):
                w_t = traj["cum_rho"][t]
                r_t = traj["disc_rewards"][t]
                weights_t.append(w_t)
                weighted_rewards_t.append(w_t * r_t)

        if len(weights_t) == 0:
            continue

        weights_t = np.asarray(weights_t, dtype=np.float64)
        weighted_rewards_t = np.asarray(weighted_rewards_t, dtype=np.float64)

        # ordinary PDIS
        pdis += float(np.mean(weighted_rewards_t))

        # weighted PDIS averaged over total number of trajectories
        # (missing suffix of shorter trajs contributes 0 implicitly)
        wpdis += float(np.sum(weighted_rewards_t) / n)

        # self-normalized / cumulative weighted PDIS
        cwpdis += float(np.sum(weighted_rewards_t) / (np.sum(weights_t) + 1e-8))

        ess_t = (np.sum(weights_t) ** 2) / (np.sum(weights_t ** 2) + 1e-8)
        ess_t_list.append(float(ess_t))

    # ESS over final clipped trajectory weights
    ess = float((np.sum(final_weights) ** 2) / (np.sum(final_weights ** 2) + 1e-8))
    ess_min = float(np.min(ess_t_list)) if ess_t_list else 0.0
    ess_mean = float(np.mean(ess_t_list)) if ess_t_list else 0.0

    if debug:
        order = np.argsort(-np.abs(final_weights))
        print("\n[OPE DEBUG] top trajectories by |final clipped weight|")

        for rank, idx in enumerate(order[:top_k]):
            dbg = ope_trajs[idx].get("debug", {})
            actions = dbg.get("actions", [])

            if len(actions) > 0:
                action_counts = {
                    int(a): int(actions.count(a))
                    for a in sorted(set(actions))
                }
                action_preview = actions[:30]
            else:
                action_counts = {}
                action_preview = []

            print("\n" + "-" * 80)
            print(f"rank={rank}")
            print(f"traj_idx={idx}")
            print(f"stay_id={dbg.get('stay_id', 'NA')}")
            print(f"final_w_clipped={final_weights[idx]:.6e}")
            print(f"final_w_raw={final_weights_raw[idx]:.6e}")
            print(f"traj_return={traj_returns[idx]:.6f}")
            print(f"T={dbg.get('T', 'NA')}")
            print(f"frac_action_0={dbg.get('frac_action_0', 'NA')}")
            print(f"num_action_0={dbg.get('num_action_0', 'NA')}/{dbg.get('num_steps', 'NA')}")
            print(f"action_counts={action_counts}")
            print(f"actions_first_30={action_preview}")
            print(f"pi_ai_min={dbg.get('pi_ai_min', 'NA'):.6e}")
            print(f"pi_ai_max={dbg.get('pi_ai_max', 'NA'):.6e}")
            print(f"pi_b_min={dbg.get('pi_b_min', 'NA'):.6e}")
            print(f"pi_b_max={dbg.get('pi_b_max', 'NA'):.6e}")
            print(f"rho_raw_max={dbg.get('rho_raw_max', 'NA'):.6e}")
            print(f"cum_rho_last={dbg.get('cum_rho_last', 'NA'):.6e}")
            print(f"cum_rho_raw_last={dbg.get('cum_rho_raw_last', 'NA'):.6e}")

    return {
        "is": is_est,
        "wis": wis,
        "pdis": pdis,
        "wpdis": wpdis,
        "cwpdis": cwpdis,
        "ess": ess,
        "ess_min": ess_min,
        "ess_mean": ess_mean,
    }

def debug_ope_summary(ope_trajs):
    if len(ope_trajs) == 0:
        print("[OPE DEBUG] no trajectories")
        return

    pi_b_mins = np.array([traj["debug"]["pi_b_min"] for traj in ope_trajs], dtype=np.float64)
    pi_ai_mins = np.array([traj["debug"]["pi_ai_min"] for traj in ope_trajs], dtype=np.float64)
    rho_raw_maxs = np.array([traj["debug"]["rho_raw_max"] for traj in ope_trajs], dtype=np.float64)

    cum_last = np.array([traj["debug"]["cum_rho_last"] for traj in ope_trajs], dtype=np.float64)
    cum_last_raw = np.array([traj["debug"]["cum_rho_raw_last"] for traj in ope_trajs], dtype=np.float64)

    print("\n[OPE DEBUG SUMMARY]")
    print("num_trajs =", len(ope_trajs))

    print("pi_b_min global min =", float(pi_b_mins.min()))
    print("pi_b_min median     =", float(np.median(pi_b_mins)))

    print("pi_ai_min global min =", float(pi_ai_mins.min()))
    print("pi_ai_min median     =", float(np.median(pi_ai_mins)))

    print("rho_raw_max global max =", float(rho_raw_maxs.max()))
    print("rho_raw_max median     =", float(np.median(rho_raw_maxs)))

    print("\n[clipped cumulative weights]")
    print("cum_rho_last min    =", float(cum_last.min()))
    print("cum_rho_last median =", float(np.median(cum_last)))
    print("cum_rho_last max    =", float(cum_last.max()))
    print("num cum_rho_last > 1e1 =", int(np.sum(cum_last > 1e1)))
    print("num cum_rho_last > 1e2 =", int(np.sum(cum_last > 1e2)))
    print("num cum_rho_last > 1e3 =", int(np.sum(cum_last > 1e3)))
    print("num cum_rho_last < 1e-3 =", int(np.sum(cum_last < 1e-3)))
    print("num cum_rho_last < 1e-6 =", int(np.sum(cum_last < 1e-6)))

    print("\n[raw cumulative weights]")
    print("cum_rho_raw_last min    =", float(cum_last_raw.min()))
    print("cum_rho_raw_last median =", float(np.median(cum_last_raw)))
    print("cum_rho_raw_last max    =", float(cum_last_raw.max()))
    print("num cum_rho_raw_last > 1e1 =", int(np.sum(cum_last_raw > 1e1)))
    print("num cum_rho_raw_last > 1e2 =", int(np.sum(cum_last_raw > 1e2)))
    print("num cum_rho_raw_last > 1e3 =", int(np.sum(cum_last_raw > 1e3)))
    print("num cum_rho_raw_last < 1e-3 =", int(np.sum(cum_last_raw < 1e-3)))
    print("num cum_rho_raw_last < 1e-6 =", int(np.sum(cum_last_raw < 1e-6)))