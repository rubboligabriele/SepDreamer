import collections
import io
import os
import json
import pathlib
import random

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd
from torch.utils.tensorboard import SummaryWriter

to_np = lambda x: x.detach().cpu().numpy()

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
    to_maximize = ["cwpdis", "ess", "mortality_decrease"]

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

    # Combine values and steps
    summary = {}
    for k in best:
        summary[k] = best[k]
        summary[f"{k}_step"] = best_steps.get(k, None)

    return summary

def from_generator(generator, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(generator))
        data = {}
        for key in batch[0].keys():
            data[key] = []
            for i in range(batch_size):
                data[key].append(batch[i][key])
            data[key] = np.stack(data[key], 0)
        yield data

def sample_episodes(episodes, length, seed=0):
    np_random = np.random.RandomState(seed)
    while True:
        size = 0
        ret = None
        p = np.array(
            [len(next(iter(episode.values()))) for episode in episodes.values()]
        )
        p = p / np.sum(p)
        while size < length:
            episode = np_random.choice(list(episodes.values()), p=p)
            total = len(next(iter(episode.values())))
            # make sure at least one transition included
            if total < 2:
                continue
            if not ret:
                index = int(np_random.randint(0, total - 1))
                ret = {
                    k: v[index : min(index + length, total)].copy()
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 'is_first' comes after 'is_last'
                index = 0
                possible = length - size
                ret = {
                    k: np.append(
                        ret[k], v[index : min(index + possible, total)].copy(), axis=0
                    )
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][size] = True
            size = len(next(iter(ret.values())))
        yield ret

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
    filenames = [f for f in os.listdir(saved_eps_dir) if f.endswith('.npz')]
    stay_ids = [fname[:-4] for fname in filenames]
    return stay_ids

def load_split_episodes(saved_eps_dir, stay_ids):
    episodes = {}
    for stay_id in stay_ids:
        filepath = os.path.join(saved_eps_dir, f"{stay_id}.npz")
        episode = load_episode_npz(filepath)
        episodes[stay_id] = episode
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
        device="cuda",
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255, device=device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

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
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        elif self._agg == "raw":
            loss = distance
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="raw", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

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
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        elif self._agg == "raw":
            loss = distance
        else:
            raise NotImplementedError(self._agg)
        return -loss

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
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
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
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if type(last) == type({}):
                outputs = {
                    key: value.clone().unsqueeze(0) for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
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
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = torch.cat(
                        [outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat(
                                [outputs[j][key], last[j][key].unsqueeze(0)], dim=0
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if type(last) == type({}):
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


def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
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
        items_to_load = torch.load(pathlib.Path(dir).expanduser() / f"agent_{epoch}.pt", map_location=device)
        agent.load_state_dict(items_to_load["agent_state_dict"])
        if actor_lr:
            items_to_load["optims_state_dict"]['_task_behavior._actor_opt._opt']['param_groups'][0]['lr'] = actor_lr
            items_to_load["optims_state_dict"]['_task_behavior._value_opt._opt']['param_groups'][0]['lr'] = value_lr
        recursively_load_optim_state_dict(agent, items_to_load["optims_state_dict"])
    else:
        checkpoint = torch.load(pathlib.Path(dir).expanduser() / f"{model_name}_{epoch}.pt", map_location=device)
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

def plot_mortality_vs_expected_return(reward_values, mortality, num_bins=20):
    # Trim to middle 98% quantile range
    q01, q99 = np.quantile(reward_values, [0.01, 0.99])
    mask = (reward_values >= q01) & (reward_values <= q99)
    reward_values = reward_values[mask]
    mortality = mortality[mask.squeeze()]

    # Define bin edges and centers
    bin_edges = np.linspace(q01, q99, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Assign each reward to a bin index
    bin_indices = np.digitize(reward_values, bin_edges) - 1  # bin index ∈ [0, num_bins-1]

    # Initialize arrays
    mortality_means = np.empty(num_bins)
    mortality_stds = np.empty(num_bins)
    mortality_means[:] = np.nan
    mortality_stds[:] = np.nan

    # Vectorized bin stats computation
    for i in range(num_bins):
        in_bin = bin_indices == i
        if np.sum(in_bin) >= 2:
            mortality_means[i] = np.mean(mortality[in_bin])
            mortality_stds[i] = sem(mortality[in_bin])

    # Filter out empty bins (NaNs)
    valid = ~np.isnan(mortality_means)
    bin_centers = bin_centers[valid]
    mortality_means = mortality_means[valid]
    mortality_stds = mortality_stds[valid]

    # Smooth (optional)
    smoothed = sliding_mean(mortality_means)
    smoothed_std = sliding_mean(mortality_stds)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor='white')
    ax.plot(bin_centers, smoothed)
    ax.fill_between(bin_centers, smoothed - smoothed_std, smoothed + smoothed_std, color='#ADD8E6')
    ax.set_xlabel("Expected Return", fontsize=15)
    ax.set_ylabel("Mortality", fontsize=15)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.tick_params(labelsize=15)
    ax.grid()
    fig.tight_layout()
    plt.show()

    return fig, bin_centers, smoothed, smoothed_std

def plot_mortality_vs_episode_return(
    episode_returns: np.ndarray,
    mortalities: np.ndarray,
    num_bins: int = 6,
):
    """
    Args:
        episode_returns: shape [N], float, per-trajectory return
        mortalities: shape [N], binary mortality (1 = died, 0 = survived)
        num_bins: how many bins to split returns into
        smooth: whether to apply LOWESS smoothing (optional)

    Returns:
        fig, bin_centers, mortality_means, mortality_sems
    """
    assert len(episode_returns) == len(mortalities), "Mismatched input lengths"
    
    # Bin by return quantiles
    quantile_bins = np.quantile(episode_returns, np.linspace(0, 1, num_bins + 1))
    bin_centers = []
    mortality_means = []
    mortality_sems = []

    for i in range(num_bins):
        lower = quantile_bins[i]
        upper = quantile_bins[i + 1]
        in_bin = (episode_returns >= lower) & (episode_returns < upper) if i < num_bins - 1 else (episode_returns >= lower) & (episode_returns <= upper)
        if np.sum(in_bin) == 0:
            continue
        bin_returns = episode_returns[in_bin]
        bin_mortality = mortalities[in_bin]
        bin_centers.append(bin_returns.mean())
        mortality_means.append(bin_mortality.mean())
        mortality_sems.append(sem(bin_mortality))

    bin_centers = np.array(bin_centers)
    mortality_means = np.array(mortality_means)
    mortality_sems = np.array(mortality_sems)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        bin_centers,
        mortality_means,
        yerr=mortality_sems,
        fmt="-o",
        capsize=4,
        label="Mortality rate",
    )
    ax.set_xlabel("Episode Return")
    ax.set_ylabel("Mortality Rate")
    ax.set_title("Mortality vs. Episode Return")
    ax.grid(True)
    ax.legend()

    return fig, bin_centers, mortality_means, mortality_sems

def calculate_esmitated_mortality(ai_values, bin_centers, smoothed, smoothed_sem):
    q_mean = np.mean(ai_values)
    idx = np.argmin(np.abs(bin_centers - q_mean))
    ai_mortality = smoothed[idx]
    ai_std = smoothed_sem[idx]
    return ai_mortality, ai_std

def cwpdis_ess_eval(ai_action, phys_action, reward, gamma=0.99):
    # Convert one-hot or logits to discrete action indices
    ai_action = np.argmax(ai_action, axis=-1)     # [T] or [B, T]
    phys_action = np.argmax(to_np(phys_action), axis=-1) # [T] or [B, T]

    # Flatten everything to [T] if needed
    ai_action = ai_action.flatten()
    phys_action = phys_action.flatten()
    reward = reward.flatten()

    # Concordance: whether AI action matches clinician action
    concordant = (ai_action == phys_action)

    v_cwpdis = 0.0
    ess = 0.0

    for t in range(len(concordant)):
        if concordant[t]:
            v_cwpdis += (gamma ** (t + 1)) * reward[t]
            ess += 1

    return v_cwpdis, ess

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
    total = targets.size(1)
    
    return round(correct / total * 100, 2)
