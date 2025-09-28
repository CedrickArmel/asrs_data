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

import pytest
import spacy
import torch
from transformers import AutoTokenizer

from asrsclassifier.data.datasets import ClsfierDataset


@pytest.fixture
def test_data():
    return [
        {
            "narrative": "The UAV collided with a BIRD. Bonjour , comment ça va ? Très bien , merci ! ;,, . Bonjour ; ensuite. Incroyable!! Quoi?? C'est fou!!?? Non?! ok. Bonjour ?!;;!! ok. Bonjour?!;;!!",
            "anomaly": "ATC Issue; Aircraft Equipment; Airspace Violation ;Conflict;Deviation - Altitude;Deviation - Speed;Deviation - Track/Heading;Deviation/Discrepancy - Procedural;Flight Deck/Cabin/Aircraft Event;Ground Event / Encounter;Ground Excursion;Ground Incursion;Inflight Event/Encounter;No Specific Anomaly Occurred",
        },
        {"narrative": "", "anomaly": "ATC Issue; Aircraft Equipment "},
    ]


@pytest.fixture
def decoder():
    with open("data/01_primary/abs_decoder.json", "r") as f:
        decoder = json.load(f)
    return decoder


@pytest.fixture
def mapper():
    with open("data/01_primary/one_hot_mapping.json", "r") as f:
        mapper = json.load(f)
    return mapper


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)


@pytest.fixture
def nlp():
    return spacy.blank("en")


def test_len(test_data, tokenizer, decoder, mapper, nlp):
    dataset = ClsfierDataset(
        tokenizer, test_data, max_len=16, decoder=decoder, lang=nlp, mapper=mapper
    )
    assert len(dataset) == 2


def test_getitem(test_data, tokenizer, decoder, mapper, nlp):
    dataset = ClsfierDataset(
        tokenizer, test_data, max_len=512, decoder=decoder, lang=nlp, mapper=mapper
    )
    item = dataset[0]
    assert set(item.keys()) == set(["input", "target"])
    assert set(item["input"].keys()) == set(
        ["input_ids", "attention_mask", "token_type_ids"]
    )
    assert isinstance(item["input"]["input_ids"], torch.Tensor)
    assert isinstance(item["input"]["attention_mask"], torch.Tensor)
    assert isinstance(item["input"]["token_type_ids"], torch.Tensor)
    assert item["input"]["input_ids"].shape[0] == 512
    assert item["target"].shape[0] == len(mapper)


def test_decode_abs(test_data, tokenizer, decoder, mapper, nlp):
    dataset = ClsfierDataset(
        tokenizer, test_data, max_len=16, decoder=decoder, lang=nlp, mapper=mapper
    )
    text = test_data[0]["narrative"]
    decoded = dataset._decode_abs(text=text)
    print(decoded)
    print(
        "The Unmanned Aerial Vehicle collided with a BIRD. Bonjour , comment ça va ? Très bien , merci ! ;,, . Bonjour ; ensuite. Incroyable!! Quoi?? C'est fou!!?? Non?! ok. Bonjour ?!;;!! ok. Bonjour?!;;!!"
    )
    assert (
        "The Unmanned Aerial Vehicle collided with a BIRD. Bonjour , comment ça va ? Très bien , merci ! ;,, . Bonjour ; ensuite. Incroyable!! Quoi?? C'est fou!!?? Non?! ok. Bonjour ?!;;!! ok. Bonjour?!;;!!"
        == decoded
    )


def test_clean_punc(test_data, tokenizer, decoder, mapper, nlp):
    from asrsclassifier.data.datasets import ClsfierDataset

    dataset = ClsfierDataset(
        tokenizer, test_data, max_len=16, decoder=decoder, lang=nlp, mapper=mapper
    )
    text = test_data[0]["narrative"]
    cleaned = dataset._clean_punc(text=text)
    assert (
        "The UAV collided with a BIRD. Bonjour, comment ça va? Très bien, merci! Bonjour; ensuite. Incroyable! Quoi? C'est fou? Non? ok. Bonjour? ok. Bonjour?"
        == cleaned
    )


def test_one_hot(test_data, tokenizer, decoder, mapper, nlp):
    from asrsclassifier.data.datasets import ClsfierDataset

    dataset = ClsfierDataset(
        tokenizer, test_data, max_len=16, decoder=decoder, lang=nlp, mapper=mapper
    )
    text1 = test_data[0]["anomaly"]
    target1 = dataset._one_hot(text1)
    text2 = test_data[1]["anomaly"]
    target2 = dataset._one_hot(text2)
    assert target1 == [1] * 14
    assert target2 == [1] * 2 + [0] * 12
