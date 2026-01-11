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
import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn.init import (
    calculate_gain,
    kaiming_normal_,
    kaiming_uniform_,
    xavier_normal_,
    xavier_uniform_,
)

from .pylogger import RankedLogger
from .rich_utils import enforce_tags, print_config_tree

log = RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:  # type: ignore[attr-defined]
                    log.info("Closing wandb!")
                    wandb.finish()  # type: ignore[attr-defined]

        return metric_dict, object_dict

    return wrap


def get_metric_value(
    metric_dict: Dict[str, Any], metric_name: Optional[str]
) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def get_seeded_generator(seed: "int") -> "torch.Generator":
    NP_MAX = np.iinfo(np.uint32).max
    MAX_SEED = NP_MAX + 1
    seed = seed % MAX_SEED
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def balance_data_world_size(data: list[dict], batch: int, worldsize: int) -> list[dict]:
    """Complete `data` by randomly sampling a set of observations from data.
    Usefull to avoid `drop_last` in data loader and train/evaluate on whole data.

    Args:
        data (list[dict]): The data to complete.
        batch (int): Batch size.
        worldsize (int): Worldsize

    Returns:
        list[dict]: Completed data
    """

    if (x := len(data) % (batch * worldsize)) != 0:
        data += random.choices(population=data, k=((batch * worldsize) - x))
    return data


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
