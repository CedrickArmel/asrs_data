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
from functools import partial
from typing import Any

import lightning.pytorch as L
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class ASRSClassifier(L.LightningModule):
    """ASRSClassifier Lightning Module"""

    def __init__(
        self,
        net: Module,
        loss: Module,
        metric: Metric,
        optimizer: partial[Optimizer],
        scheduler: partial[LRScheduler],
        trainable_layers: dict[str, list],
        priors: list[float] | None,
    ) -> None:
        super().__init__()
        self.net = net
        self.loss = loss
        self.partial_optimizer = optimizer
        self.trainable_layers = trainable_layers
        self.partial_metric = metric
        self.partial_scheduler = scheduler
        self.priors = torch.tensor(priors) if priors is not None else priors

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`."""
        return self.net(**inputs)

    def set_trainable(self):
        if self.trainable_layers:
            for path, indices in self.trainable_layers.items():
                try:
                    parts = path.split(".")
                    submodule = self.trainer.model.net
                    for part in parts:
                        submodule = getattr(submodule, part)
                    if indices is None:
                        self._set_layer_trainable(submodule, True)
                    else:
                        for idx in indices:
                            self._set_layer_trainable(submodule[idx], True)
                except (AttributeError, IndexError, TypeError) as e:
                    print(f"[Warning] Failed to set layer '{path}': {e}")

    def configure_optimizers(self) -> "dict[str, Any] | Optimizer":  # type: ignore[override]
        """Return the optimizer and an optionnal lr_scheduler"""
        optimizer: Optimizer = self.partial_optimizer(
            params=self.trainer.model.parameters()  # type: ignore[union-attr]
        )
        scheduler: LRScheduler = self.partial_scheduler(optimizer=optimizer)
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

    def setup(self, stage: "str") -> "None":
        """Called at the beginning of each stage in oder to build model dynamically."""
        if stage == "fit":
            if self.trainable_layers is not None:
                self.trainer.model.apply(  # type: ignore[union-attr]
                    lambda layer: self._set_layer_trainable(
                        layer=layer, trainable=False
                    )
                )
                self.set_trainable()
                if self.priors is not None:
                    with torch.no_grad():
                        self.trainer.model.net.classifier.bias.copy_(-torch.log(self.priors))  # type: ignore[union-attr, operator]

    def on_fit_start(self) -> "None":
        """Called at the very beginning of fit."""
        self._shared_start()

    def on_test_start(self):
        self.output = []
        self._shared_start()

    def training_step(
        self, batch: "dict[str, Any]", batch_idx: "int"
    ) -> "torch.Tensor":
        output_dict = self._model_step(batch)
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
        loss, logits = self._shared_eval_step(batch)
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
        return logits.sigmoid()

    def test_step(self, batch: "dict[str, Any]", batch_idx: "int"):
        loss, logits = self._shared_eval_step(batch)
        self.output.append(
            torch.cat(  # type: ignore[call-overload]
                [batch["acn"].cpu(), logits.sigmoid().cpu(), batch["target"].cpu()],
                axis=1,
            )
        )
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        return logits.sigmoid()

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

    # TODO: Make sure to manage this in artefacts
    def on_test_epoch_end(self) -> None:
        output: "torch.Tensor" = dim_zero_cat(x=self.output)
        torch.save(
            output,
            os.path.join(
                self.trainer.log_dir, f"preds-rank{self.trainer.global_rank}.pt"  # type: ignore[arg-type]
            ),
        )

    def _model_step(self, batch: "dict[str, Any]") -> "dict[str, Any]":
        """Perform a single model step on a batch of data."""
        has_target = "target" in batch
        outputs = {}
        if self.training:
            output = self(batch["input"])
        else:
            with torch.no_grad():
                output = self(batch["input"])
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

    def _set_layer_trainable(self, layer: torch.nn.Module, trainable: bool = False):
        for param in layer.parameters():
            param.requires_grad = trainable

    def _shared_start(self):
        if torch.distributed.is_initialized():
            if not hasattr(self, "gloo_group"):
                self.gloo_groupg = torch.distributed.new_group(backend="gloo")
                self.metric = self.partial_metric(process_group=self.gloo_groupg)
        else:
            self.metric = self.partial_metric()

    def _shared_eval_step(self, batch: "dict[str, Any]"):
        output_dict = self._model_step(batch)
        loss: "torch.Tensor" = output_dict["loss"]
        logits = output_dict["logits"]
        return loss, logits
