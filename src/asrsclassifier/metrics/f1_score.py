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

from typing import Any

import numpy as np
import torch
from torchmetrics import Metric
from torchmetrics.functional import f1_score
from torchmetrics.utilities import dim_zero_cat


class F1Sscore(Metric):
    def __init__(
        self,
        task: "str",
        num_classes: "int | None" = None,
        num_labels: "int | None" = None,
        average: "str | None" = "micro",
        multidim_average: "str | None" = "global",
        top_k: "int | None" = 1,
        ignore_index: "int | None" = None,
        validate_args: "bool" = True,
        zero_division: "float" = 0.0,
        **kwargs,
    ) -> "None":
        super().__init__(**kwargs)
        self.task = task
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.average = average
        self.multidim_average = multidim_average
        self.top_k = top_k
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division
        self.add_state(name="preds", default=[], dist_reduce_fx="cat")
        self.add_state(name="targets", default=[], dist_reduce_fx="cat")
        self.preds: "list[torch.Tensor]"
        self.targets: "list[torch.Tensor]"

    def update(self, pred: "torch.Tensor", target: "torch.Tensor") -> "None":
        self.preds.append(pred)
        self.targets.append(target)

    def compute(self) -> "dict[str, Any]":
        preds: "torch.Tensor" = dim_zero_cat(x=self.preds)
        target: "torch.Tensor" = dim_zero_cat(x=self.targets)
        ths = np.arange(start=0.0, stop=1.0, step=0.001)
        class_scores_list = []
        scores = []
        for t in ths:
            score = (
                f1_score(
                    threshold=t,
                    preds=preds,
                    target=target,
                    task=self.task,
                    num_classes=self.num_classes,
                    num_labels=self.num_labels,
                    average=self.average,
                    multidim_average=self.multidim_average,
                    top_k=self.top_k,
                    ignore_index=self.ignore_index,
                    validate_args=self.validate_args,
                    zero_division=self.zero_division,
                )
                .cpu()
                .numpy()
            )
            class_scores_list += [score]
            scores += [score.mean()]
        scores_per_class = np.array(class_scores_list)
        best_idx_per_class = np.argmax(scores_per_class, axis=0)
        best_thd_per_class = ths[best_idx_per_class]
        best_scores_per_label = [
            scores_per_class[i, c] for c, i in enumerate(best_idx_per_class)
        ]  # optimal score @ each label best's thd
        score = np.mean(best_scores_per_label)
        std = np.std(best_scores_per_label)
        # best_idx = int(np.argmax(a=scores))
        # thd = float(ths[best_idx])  # best global threshold
        # score = float(scores[best_idx]) # score @ thd
        thd_per_label = {
            f"thd_{i}": float(v)
            for i, v in zip(range(len(best_thd_per_class)), best_thd_per_class)
        }
        scores_per_label = {
            f"score_{i}": float(v)
            for i, v in zip(range(len(best_scores_per_label)), best_scores_per_label)
        }
        # global_values = dict(thd=thd, score=score)
        global_values = dict(std=std, score=score)
        per_class_values = {**thd_per_label, **scores_per_label}
        return dict(pbar=global_values, per_class=per_class_values)
