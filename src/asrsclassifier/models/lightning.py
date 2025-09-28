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

from pathlib import Path
from typing import Any

import lightning.pytorch as L
import torch
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import wandb
from asrsclassifier.losses import FocalLoss
from asrsclassifier.metrics import F1Sscore
from asrsclassifier.utils import get_optimizer, get_scheduler

from .load_models import get_model


class ASRSClassifier(L.LightningModule):
    """ASRSClassifier Lightning Module"""

    def __init__(self, cfg: "DictConfig") -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = get_model(**cfg.models)
        self.loss = FocalLoss(**cfg.focal_loss)

    def forward(self, batch: "dict[str, Any]") -> "dict[str, Any]":
        """Perform a forward pass through the model."""
        has_target = "target" in batch
        outputs = {}
        if self.training:
            output = self.backbone(**batch["input"])
        else:
            with torch.no_grad():
                output = self.backbone(**batch["input"])
        try:
            logits = output.logits
        except AttributeError:
            logits = output

        if self.training:
            loss = self.loss(input=logits, target=batch["target"])
        else:
            if has_target:
                with torch.no_grad():
                    loss = self.loss(input=logits, target=batch["target"])
        if has_target:
            outputs["loss"] = loss
        if not self.training:
            outputs["logits"] = logits
        return outputs

    def _set_layer_trainable(self, layer: "torch.nn.Layer", trainable: "bool" = False):
        for param in layer.parameters():
            param.requires_grad = trainable

    def set_trainable(self):
        if self.trainable_layers:
            for path, indices in self.trainable_layers.items():
                try:
                    parts = path.split(".")
                    submodule = self.backbone
                    for part in parts:
                        submodule = getattr(submodule, part)
                    if indices is None:
                        self._set_layer_trainable(submodule, True)
                    else:
                        for idx in indices:
                            self._set_layer_trainable(submodule[idx], True)
                except (AttributeError, IndexError, TypeError) as e:
                    print(f"[Warning] Failed to set layer '{path}': {e}")

    def setup(self, stage: "str") -> "None":
        """Called at the beginning of each stage in oder to build model dynamically."""
        # if self.trainer.is_global_zero:
        #     self.run = wandb.run
        # self._logged_model_time: "dict[str, float]" = {}
        # self._checkpoint_name: "str | None" = None
        self.trainable_layers = self.cfg.training.supervision.trainable_layers
        stepping_batches = self.trainer.estimated_stepping_batches
        max_epochs = self.cfg.trainers.lightning.max_epochs
        world_size = self.trainer.world_size

        self.cfg.lr *= self.trainer.world_size
        self.training_steps = stepping_batches * max_epochs * world_size

        if stage == "fit":
            if self.trainable_layers is not None:
                self.backbone.apply(
                    lambda layer: self._set_layer_trainable(
                        layer=layer, trainable=False
                    )
                )
                self.set_trainable()

    def configure_optimizers(self) -> "dict[str, Any] | Optimizer":
        """Return the optimizer and an optionnal lr_scheduler"""
        optimizer: "Optimizer" = get_optimizer(self.backbone, **self.cfg.optimizer)
        scheduler: "LRScheduler | None" = get_scheduler(
            optimizer=optimizer,
            training_steps=self.training_steps,
            **self.cfg.scheduler,
        )
        return (
            dict(
                optimizer=optimizer,
                lr_scheduler=dict(
                    scheduler=scheduler, interval="step", frequency=1, name="lr"
                ),
            )
            if scheduler is not None
            else optimizer
        )

    def training_step(
        self, batch: "dict[str, Any]", batch_idx: "int"
    ) -> "torch.Tensor":
        output_dict = self(batch)
        loss: "torch.Tensor" = output_dict["loss"]
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch: "dict[str, Any]", batch_idx: "int"
    ) -> "torch.Tensor":
        """Operates on a single batch of data from the validation set"""
        output_dict = self(batch)
        loss: "torch.Tensor" = output_dict["loss"]
        logits = output_dict["logits"]
        self.metric.update(logits.sigmoid(), batch["target"])
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        return logits

    def on_fit_start(self) -> "None":
        """Called at the very beginning of fit."""
        if torch.distributed.is_initialized():
            if not hasattr(self, "gloo_group"):
                self.gloo_groupg = torch.distributed.new_group(backend="gloo")
                self.metric = F1Sscore(
                    process_group=self.gloo_groupg,
                    **self.cfg.metrics.f1_score.init,
                    **self.cfg.metrics.f1_score.super,
                )
        else:
            self.metric = F1Sscore(
                **self.cfg.metrics.f1_score.init, **self.cfg.metrics.f1_score.super
            )

    def on_train_batch_end(self, outputs, batch, batch_idx) -> "None":
        params = [p for p in self.parameters() if p.grad is not None]
        if len(params) == 0:
            total_norm = torch.tensor(0.0)
        else:
            total_norm = torch.norm(
                torch.cat([p.detach().view(-1) for p in params]),
                p=2,
            )
        self.log(
            "weight_norm",
            total_norm,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> "None":
        """Called after the epoch ends to agg preds and logging"""
        metrics = self.metric.compute()
        self.log_dict(
            metrics["pbar"],
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log_dict(
            metrics["per_class"],
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.metric.reset()

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: "float | None" = None,
        gradient_clip_algorithm: "Any | None" = None,
    ) -> "None":
        """Gradient clipping and tracking before/afater clipping"""
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if len(grads) == 0:
            total_norm_before = torch.tensor(0.0)
        else:
            total_norm_before = torch.norm(
                torch.cat([g.detach().view(-1) for g in grads]),
                p=2,
            )

        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if len(grads) == 0:
            total_norm_after = torch.tensor(0.0)
        else:
            total_norm_after = torch.norm(
                torch.cat([g.detach().view(-1) for g in grads]),
                p=2,
            )

        log_dict: "dict[str, torch.Tensor]" = dict(
            grad_norm=total_norm_before, clip_grad_norm=total_norm_after
        )

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
        )

    def _scan_and_log_checkpoints(self) -> None:
        checkpoint_callback = self.trainer.checkpoint_callback
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)
        for t, p, s, _ in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, torch.Tensor) else s,
                "original_filename": Path(p).name,
                checkpoint_callback.__class__.__name__: {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                    ]
                    if hasattr(checkpoint_callback, k)
                },
            }
            if not self._checkpoint_name:  # type: ignore[has-type]
                self._checkpoint_name = f"model-{self.run.id}"
            artifact = wandb.Artifact(  # type: ignore[attr-defined]
                name=self._checkpoint_name, type="model", metadata=metadata
            )
            artifact.add_file(p, name=f"{self._checkpoint_name}.ckpt")
            aliases = (
                ["latest", "best"]
                if p == checkpoint_callback.best_model_path
                else ["latest"]
            )
            if self.trainer.is_global_zero:
                self.run.log_artifact(artifact, aliases=aliases)
            self._logged_model_time[p] = t
