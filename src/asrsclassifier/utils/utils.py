# MIT License
#
# Copyright (c) 2025, Yebouet Cédrick-Armel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import random
from typing import Any
from warnings import warn

import numpy as np
import torch
import torch.optim as optim
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.profilers import Profiler, PyTorchProfiler, XLAProfiler
from torch.nn.init import (
    calculate_gain,
    kaiming_normal_,
    kaiming_uniform_,
    xavier_normal_,
    xavier_uniform_,
)
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    MultiStepLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from wonderwords import RandomWord


def create_milestones(steps: "int", m: "int") -> "list[int]":
    """returns a list of milestones for the given number of steps and m."""
    g = int(steps // m)
    milestones = []
    for i in range(1, m + 1):
        milestones += [i] * g
    return milestones


def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def get_callbacks(callbacks_args: "dict[str, Any]") -> "tuple[Callback, ...]":
    chckpt_cb = ModelCheckpoint(**callbacks_args["checkpoint"])
    lr_cb = LearningRateMonitor(**callbacks_args["lr_monitor"])
    return chckpt_cb, lr_cb


def get_data_loader(dataset, seed: "int", **kwargs) -> "DataLoader":
    g = get_seeded_generator(seed)
    loader = DataLoader(
        dataset=dataset,
        worker_init_fn=seed_worker,
        generator=g,
        **kwargs,
    )
    return loader


def get_optimizer(
    model: "torch.nn.Module",
    name: "str" = "SGD",
    **kwargs,
) -> "optim.Optimizer | None":

    params_ = model.parameters()
    args = kwargs[name.lower()]

    if name == "Adam":
        optimizer: "optim.Optimizer" = optim.Adam(params_, **args)

    elif name == "AdamW_plus":
        nparams_ = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        lr = args["lr"]
        weight_decay = args["weight_decay"]
        params = [
            {
                "params": [
                    param
                    for name, param in nparams_
                    if (not any(nd in name for nd in no_decay))
                ],
                "lr": lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in nparams_
                    if (any(nd in name for nd in no_decay))
                ],
                "lr": lr,
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(params, lr=lr)

    elif name == "AdamW":
        optimizer = optim.AdamW(params_, **args)

    elif name == "SGD":
        optimizer = optim.SGD(params_, **args)
    return optimizer


def get_profiler(accelerator: "str" = "xla", **kwargs) -> "Profiler | None":
    """Returns a suitable profiler for the used accelerator"""
    args = kwargs[accelerator]
    if accelerator == "xla":
        profiler = XLAProfiler(**args)
    elif accelerator == "cuda":
        profiler = PyTorchProfiler(**args)
    else:
        profiler = None
    return profiler


def get_seeded_generator(seed: "int") -> "torch.Generator":
    NP_MAX = np.iinfo(np.uint32).max
    MAX_SEED = NP_MAX + 1
    seed = seed % MAX_SEED
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_scheduler(
    optimizer: "optim.Optimizer",
    training_steps: "int",
    name: "str" = "cosine_wr",
    warmup: "int" = 100,
    **kwargs,
) -> "LRScheduler | None":
    """
    Creates and returns a learning rate scheduler based on the provided configuration.
    Returns:
        LRScheduler: The configured learning rate scheduler or None if no valid scheduler is specified.
    """

    if name not in ["multistep", "cosine", "cosine_wr", "constant", "none"]:
        raise ValueError(
            f"Invalid schedule type: {name}. Supported types are 'multistep', 'cosine', 'cosine_wr', 'constant', 'none'."
        )
    args = kwargs[name]
    if name == "multistep":
        steps: "int" = training_steps - warmup
        milestones = range(1, steps, (steps // args.milestones))
        scheduler = MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=args.gamma
        )

    elif name == "cosine":
        scheduler = CosineAnnealingLR(optimizer=optimizer, **args)

    elif name == "cosine_wr":
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, **args)
    elif name == "constant":
        scheduler = ConstantLR(optimizer=optimizer, **args)
    elif name == "none":
        if warmup > 0:
            warn(
                "Warmup is set to a value greater than 0, but `none` is provided as schedule type."
                "Considering only the linear warmup phase. Set `warmup` to 0 to disable it."
            )
        scheduler = None

    else:
        if warmup > 0:
            warn(
                "Warmup is set to a value greater than 0, but an unsupported schedule"
                f"type: {name} is provided by user."
                "Considering only the linear warmup phase. Set `warmup` to 0 to disable it."
            )
        scheduler = None

    if warmup > 0:
        args = kwargs["linear"]
        warmup_scheduler = LinearLR(optimizer=optimizer, **args)
        if scheduler is None:
            sequential: "LinearLR" = warmup_scheduler
        else:
            sequential = SequentialLR(
                optimizer=optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup],
            )
    else:
        sequential = scheduler
    return sequential


def get_slug():
    w = RandomWord()
    return "-".join(
        [
            w.word(include_categories=["adjectives"]),
            w.word(include_categories=["nouns"]),
        ]
    )


def initialize_weights(
    module: "torch.nn.Module",
    dist: "str",
    a: "float" = 0.1,
    mode: "str" = "fan_out",
    nonlinearity: "str" = "leaky_relu",
) -> "None":
    """Applies to a model to init its params"""
    if isinstance(module, torch.nn.Linear):
        gain = calculate_gain(nonlinearity=nonlinearity)
        if dist == "normal":
            xavier_normal_(tensor=module.weight, gain=gain)
        elif dist == "uniform":
            xavier_uniform_(tensor=module.weight, gain=gain)
    elif isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
        if dist == "normal":
            kaiming_normal_(
                tensor=module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
        elif dist == "uniform":
            kaiming_uniform_(
                tensor=module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(
    seed: "int | None" = 4294967295,
    cudnn_backend: "bool" = False,
    use_deterministic_algorithms: "bool" = False,
    warn_only: "bool" = True,
) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    NP_MAX = np.iinfo(np.uint32).max
    MAX_SEED = NP_MAX + 1

    if seed is None:
        seed_ = torch.default_generator.seed() % MAX_SEED
        torch.manual_seed(seed_)
    else:
        seed = int(seed) % MAX_SEED
        torch.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

    if seed is not None and cudnn_backend:
        torch.backends.cudnn.deterministic = True  # if True, causes cuDNN to only use deterministic convolution algorithms
        torch.backends.cudnn.benchmark = False  # If True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest

    if use_deterministic_algorithms:
        torch.use_deterministic_algorithms(
            mode=use_deterministic_algorithms, warn_only=warn_only
        )  # Sets whether PyTorch operations must use “deterministic” algorithms
