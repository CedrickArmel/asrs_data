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

import tempfile

import pandas as pd
import pytest

from asrsclassifier.data.load_data import get_data


@pytest.fixture
def parquet_data():
    df = pd.DataFrame(
        {
            "acn": [1, 2, 3, 4],
            "narrative": ["a", "b", "c", "d"],
            "anomaly": [0, 1, 0, 1],
            "fold": [0, 1, 0, 1],
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name)
        yield f.name


def test_get_data_fit_mode(parquet_data):
    train, val = get_data(parquet_data, fold=0, mode="fit")
    assert isinstance(train, list)
    assert isinstance(val, list)
    assert all("acn" in record for record in train + val)
    assert all("narrative" in record for record in train + val)
    assert all("anomaly" in record for record in train + val)
    assert all(isinstance(record, dict) for record in train + val)
    assert all(
        record["acn"] != 1 and record["acn"] != 3 for record in train
    )  # fold != 0


def test_get_data_test_mode(parquet_data):
    (data,) = get_data(parquet_data, mode="test")
    assert isinstance(data, list)
    assert all("acn" in record for record in data)
    assert all("narrative" in record for record in data)
    assert all(isinstance(row, dict) for row in data)


def test_get_data_invalid_mode(parquet_data):
    with pytest.raises(ValueError):
        get_data(parquet_data, fold=0, mode="invalid")
