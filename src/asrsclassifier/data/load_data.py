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

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import spacy
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from asrsclassifier.utils import get_data_loader

from .datasets import ClsfierDataset


def get_data(
    path: "str",
    val_path: "str | None" = None,
    fold: "int" = -1,
    mode: "str" = "kfold",
) -> "Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]":

    if mode not in ["kfold", "fit", "test"]:
        raise ValueError("mode argument must be one of kfold, fit or test!")

    if not isinstance(fold, int) and fold < -1:
        raise ValueError("`fold` must be an integer >= -1.")

    if fold is None and mode == "kfold":
        raise ValueError("`fold` must be an provided `mode='kfold'` >= -1.")

    if mode in ["kfold", "fit"]:
        df = pd.read_parquet(path)
        if mode == "kfold":
            train_data = df.loc[
                df.fold != fold, ["acn", "narrative", "anomaly"]
            ].to_dict(orient="records")
            val_data = df.loc[df.fold == fold, ["acn", "narrative", "anomaly"]].to_dict(
                orient="records"
            )

        elif mode == "fit":
            train_data = df[["acn", "narrative", "anomaly"]].to_dict(orient="records")
            if val_path is not None:
                df = pd.read_parquet(val_path)
                val_data = df[["acn", "narrative", "anomaly"]].to_dict(orient="records")
            else:
                val_data = None

    else:
        train_data = None
        df = pd.read_parquet(val_path)
        val_data = df[["acn", "narrative", "anomaly"]].to_dict(orient="records")

    return train_data, val_data


def get_decoders(mapper_path: "str", decoder_path: "str"):
    with open(mapper_path, "r") as f:
        mapper = json.load(f)
    with open(decoder_path, "r") as f:
        decoder = json.load(f)
    return mapper, decoder


def load_data(cfg: "DictConfig") -> "Tuple[DataLoader, Optional[DataLoader]]":
    mapper, decoder = get_decoders(**cfg.data.decoders)
    tokenizer = AutoTokenizer.from_pretrained(cfg.models.encoder_name, use_fast=True)
    nlp = spacy.load("en_core_web_sm")
    train_data, val_data = get_data(**cfg.data.datasets)
    train_loader = None
    val_loader = None

    if train_data:
        train_ds = ClsfierDataset(
            data=train_data,
            tokenizer=tokenizer,
            mapper=mapper,
            decoder=decoder,
            lang=nlp,
            **cfg.data.params,
        )
        train_loader = get_data_loader(
            dataset=train_ds, seed=cfg.determinism.seed, **cfg.loader.train
        )

    if val_data is not None:
        bs = cfg.loader.eval.batch_size
        if (x := len(val_data) % (bs * 8)) != 0:
            val_data += val_data[
                -((bs * 8) - x) :
            ]  # complete de data so that we avoid drop last in data_loader. 8 -> 8 core (4x2 cores, ...)
        val_ds = ClsfierDataset(
            data=val_data,
            tokenizer=tokenizer,
            mapper=mapper,
            decoder=decoder,
            lang=nlp,
            **cfg.data.params,
        )
        val_loader = get_data_loader(
            dataset=val_ds, seed=cfg.determinism.seed, **cfg.loader.eval
        )
    return train_loader, val_loader
