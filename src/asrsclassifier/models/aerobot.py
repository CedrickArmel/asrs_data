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

import torch
from torch import nn
from transformers import AutoModel


class AeroBOTSequenceClassification(nn.Module):
    "AeroBOT with Classification head"

    def __init__(self, encoder_name: "str", num_labels: "int"):
        super(AeroBOTSequenceClassification, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.dense = nn.Linear(
            in_features=self.encoder.config.hidden_size, out_features=32
        )
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(32, num_labels)

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        token_type_ids: "torch.Tensor | None" = None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_token = outputs.last_hidden_state[:, 0, :]  # Extract the CLS token
        x = self.dense(cls_token)
        x = self.relu(x)
        logits = self.classifier(x)
        return logits
