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

import re
from typing import Any

import spacy
import torch
from torch.utils.data import Dataset


class ClsfierDataset(Dataset):
    """Custom dataset for ASRS anomalies classification"""

    def __init__(
        self,
        tokenizer,
        data: "list[dict[str, str]]",
        max_len: "int",
        decoder: "dict[str, str]",
        lang: "spacy.Language",
        mapping: "dict[str, int]",
        decode: "bool" = True,
        stopwords: "bool" = False,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.lang = lang
        self.mapping = mapping
        self.decode = decode
        self.stopwords = stopwords
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: "int") -> "dict[str, Any]":
        event = self.data[idx]
        narrative = self._process_single_narrative(event["narrative"])
        target = self._one_hot(text=event["anomaly"])
        tokens = self.tokenizer(
            narrative,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )
        item = {
            "input": dict(
                ids=torch.tensor(tokens["input_ids"], dtype=torch.long),
                mask=torch.tensor(tokens["attention_mask"], dtype=torch.long),
                token_type_ids=torch.tensor(tokens["token_type_ids"], dtype=torch.long),
            ),
            "target": torch.tensor(target, dtype=torch.float),
        }
        return item

    def _build_decoder_pattern(self) -> "re.Pattern":
        terms = set()
        for key in self.decoder:
            terms.update(
                {key, key.lower(), key.capitalize()} if key.isupper() else {key}
            )
        escaped_terms = [r"(?<!\w)" + re.escape(term) + r"(?!\w)" for term in terms]
        return re.compile(r"(" + "|".join(escaped_terms) + r")")

    def _clean_punc(self, text: "str"):
        t = re.sub(r"([a-zA-Z0-9])\s+([.,!?;:])", r"\1\2", text)
        t = text = re.sub(r"(\w)([!?])\2+", r"\1\2", t)
        t = re.sub(r"(\w)[!?]{2,}", r"\1?", t)
        t = re.sub(r"([.,;!?])[.,;!?]{2,}", r"\1", t)
        t = re.sub(r"(^|\s)[.,;!?](?=\s|$)", r"\1", t)
        t = re.sub(r"^[.,;!?]\s+", "", t)
        clned_text = re.sub(r"\s{2,}", " ", t).strip()
        return clned_text

    def _decode_abs(self, text: "str"):
        pattern = self._build_decoder_pattern()
        matched_abbs = [
            (abb.upper() if (abb.istitle() or abb.islower()) else abb)
            for abb in set(pattern.findall(text.strip().replace(" / ", "/")))
        ]
        for abb in matched_abbs:
            text = re.sub(
                r"(?<![-\w/'?!])" + re.escape(abb) + r"(?![-\w/'?!])",
                self.decoder[abb],
                text,
            )
        return text

    def _one_hot(self, text: "str"):
        num_labels = [0] * len(self.mapping)
        labels = [label.strip().replace(" / ", "/") for label in text.split(";")]
        for label in labels:
            num_labels[self.mapping[label]] = 1
        return num_labels

    def _remove_stopwords(self, text: "str"):
        pattern = re.compile(r"\s+([.,!?;:])")
        docs = self.lang(text)
        filtered = " ".join([token.text for token in docs if not token.is_stop])
        filtered = pattern.sub(r"\1", filtered)
        return filtered

    def _process_single_narrative(self, text: "str"):
        result = self._remove_stopwords(text=text) if not self.stopwords else text
        result = self._decode_abs(text=result) if self.decode else result
        return (
            self._clean_punc(text=result)
            if self.decode or not self.stopwords
            else result
        )
