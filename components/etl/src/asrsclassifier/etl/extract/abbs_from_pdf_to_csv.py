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
import os
from glob import glob
from pathlib import Path

import camelot
import pandas as pd

# TODO: Docstring
# TODO: Tests


def extract_abbs(src: str, dst: str, fmt: str) -> None:
    """Extracts abbrevaitions from a PDF file.

    Args:
        src (str): Source PDF file containing the abbrevaitions.
        dst (str): Destination folder.
        fmt (str): Output formats: csv, ...
    """

    tables = camelot.read_pdf(src, pages="all")
    tables.export(os.path.join(dst, Path(src).stem + f".{fmt}"), f=fmt, compress=False)


def concat_abbs_files(files: str, output: str) -> None:
    df = pd.concat(
        [pd.read_csv(f, on_bad_lines="warn") for f in glob(files)], ignore_index=True
    )
    df = (
        df.drop_duplicates(subset=["code", "description"])
        .reset_index(drop=True)
        .dropna()
    )
    df.to_csv(path_or_buf=output, index=False)


def compute_decoder_json(abbs_path: str, output_dir: str) -> None:
    df = pd.read_csv(abbs_path)
    df = df.sort_values(by="code")
    decode_dict = {k: v for (k, v) in df.to_dict(orient="split")["data"]}
    with open(os.path.join(output_dir, "abs_decoder.json"), "w") as f:
        json.dump(decode_dict, f, indent=4)


def compute_encoder_json(abbs_path: str, output_dir: str) -> None:
    df = pd.read_csv(abbs_path)
    df = df.sort_values(by="description")
    encode_dict = {k: v for (v, k) in df.to_dict(orient="split")["data"]}
    with open(os.path.join(output_dir, "abs_encoder.json"), "w") as f:
        json.dump(encode_dict, f, indent=4)
