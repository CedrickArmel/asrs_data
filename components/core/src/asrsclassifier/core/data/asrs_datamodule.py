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

from typing import Any, Dict, Optional

import hydra
import lightning.pytorch as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from asrsclassifier.core.utils import (
    balance_data_world_size,
    get_seeded_generator,
    seed_worker,
)


class ASRSDataModule(L.LightningDataModule):
    """`LightningDataModule` for the ASRS dataset.
    # TODO: Décrire brièvement les données ici

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        load: DictConfig,
        dataset: DictConfig,
        train_loader: DictConfig,
        eval_loader: DictConfig,
        seed: int,
    ) -> None:
        """Initialize a `ASRSDataModule`.
        # TODO: docstring
        """
        super().__init__()
        self.load_cfg = load
        self.dataset_cfg = dataset
        self.train_loader_cfg = train_loader
        self.eval_loader_cfg = eval_loader
        self.data_train: Optional[Dataset] = None
        self.data_eval: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_eval:
            train_data, val_data = hydra.utils.call(self.load_cfg)
            if val_data is not None:
                val_data = balance_data_world_size(
                    data=val_data,
                    batch=self.eval_loader_cfg.batch_size,
                    worldsize=self.trainer.world_size,
                )

            if train_data is not None:
                train_data = balance_data_world_size(
                    data=train_data,
                    batch=self.eval_loader_cfg.batch_size,
                    worldsize=self.trainer.world_size,
                )

            if self.hparams.mode == "train":
                self.data_train = hydra.utils.instantiate(
                    self.dataset_cfg, data=train_data
                )

                if val_data is not None:
                    self.data_eval = hydra.utils.instantiate(
                        self.dataset_cfg, data=val_data
                    )

            else:
                if val_data is not None:
                    self.data_eval = hydra.utils.instantiate(
                        self.dataset_cfg, data=val_data
                    )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return hydra.utils.instantiate(
            self.train_loader_cfg,
            dataset=self.data_train,
            worker_init_fn=seed_worker,
            generator=get_seeded_generator(self.hparams.seed),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return hydra.utils.instantiate(
            self.eval_loader_cfg,
            dataset=self.data_eval,
            worker_init_fn=seed_worker,
            generator=get_seeded_generator(self.hparams.seed),
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return hydra.utils.instantiate(
            self.eval_loader_cfg,
            dataset=self.data_eval,
            worker_init_fn=seed_worker,
            generator=get_seeded_generator(self.hparams.seed),
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
