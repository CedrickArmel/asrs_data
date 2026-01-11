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

# mypy: disable-error-code="misc, assignment, attr-defined, arg-type"

# TODO: fix ClsfierDataset arg-type with  outputs from get_data


import os

import hydra
import wandb
from asrsclassifier.config import path, wandbgroup
from asrsclassifier.data import load_data
from asrsclassifier.models import ASRSClassifier
from asrsclassifier.trainers import get_lightning_trainer
from asrsclassifier.utils import (
    flatten_dict,
    get_callbacks,
    get_profiler,
    set_seed,
)
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("eval", resolver=eval, replace=True)
OmegaConf.register_new_resolver("wandbgroup", resolver=wandbgroup, replace=True)
OmegaConf.register_new_resolver("path", resolver=path, replace=True)


# TODO: make the fit function to retrieve the correct version to resume on.
@hydra.main(version_base=None, config_path="./config", config_name="config")
def fit(cfg: "DictConfig") -> "None":

    load_dotenv()

    set_seed(**cfg.determinism)

    train_loader, val_loader = load_data(cfg)

    run_config = flatten_dict(OmegaConf.to_container(cfg, resolve=True), sep="-")
    os.makedirs(cfg.trainers.lightning.default_root_dir, exist_ok=True)
    chckpt_cb, lr_cb = get_callbacks(cfg.callbacks)

    model = ASRSClassifier(cfg)
    profiler = get_profiler(**cfg.profiler)
    logger = TensorBoardLogger(**cfg.loggers.tensorboard)

    wandb.tensorboard.unpatch()
    wandb.tensorboard.patch(root_logdir=logger.log_dir)
    wandb.init(config=run_config, **cfg.loggers.wandb, sync_tensorboard=True)

    callbacks = (
        [chckpt_cb, lr_cb, ModelSummary()]
        if cfg.trainers.callable.callbacks
        else cfg.trainers.callable.callbacks
    )
    trainer = get_lightning_trainer(
        logger=logger, callbacks=callbacks, profiler=profiler, **cfg.trainers.lightning
    )

    try:
        if cfg.training_mode == "kfold":
            trainer.fit(
                model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=cfg.training.ckpt_path,
            )
            trainer.test(model=model, dataloaders=val_loader, ckpt_path="best")

        elif cfg.training_mode == "fit":
            trainer.fit(
                model=model,
                train_dataloaders=train_loader,
                ckpt_path=cfg.training.ckpt_path,
            )
            if val_loader:
                trainer.test(model=model, dataloaders=val_loader, ckpt_path="last")
        wandb.finish()
    finally:
        wandb.finish()


if __name__ == "__main__":
    fit()
