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


def get_data(
    path: "str", fold: "int | None" = None, mode: "str" = "fit"
) -> "Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]":

    if mode not in ["fit", "test"]:
        raise ValueError("mode argument must be one of fit or test!")

    elif mode == "fit":
        df = pd.read_parquet(path)
        if isinstance(fold, int) and fold > -1:
            train_data = df.loc[
                df.fold != fold, ["acn", "narrative", "anomaly"]
            ].to_dict(orient="records")
            val_data = df.loc[df.fold == fold, ["acn", "narrative", "anomaly"]].to_dict(
                orient="records"
            )
        else:
            train_data = df.loc[df.fold != 0, ["acn", "narrative", "anomaly"]].to_dict(
                orient="records"
            )
            val_data = df.loc[df.fold == 0, ["acn", "narrative", "anomaly"]].to_dict(
                orient="records"
            )
        data = (train_data, val_data)

    elif mode == "test":
        df = pd.read_parquet(path)
        data = (df.to_dict(orient="records"), None)
    return data


def get_decoders(mapper_path: "str", decoder_path: "str"):
    with open(mapper_path, "r") as f:
        mapper = json.load(f)
    with open(decoder_path, "r") as f:
        decoder = json.load(f)
    return mapper, decoder
