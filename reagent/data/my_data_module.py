# #!/usr/bin/env python3
# ---------------------------------------------------------------------
# Implements PL data module 
# Author Neeraj Jain
# --------------------------------------------------------------------

import abc
import logging
import pickle
from typing import Dict, List, NamedTuple, Optional, Tuple
import torch

from reagent.data import myloader
from reagent.data.reagent_data_module import ReAgentDataModule

logger = logging.getLogger(__name__)


class MyDataModule(ReAgentDataModule):
    def __init__(self, train_path, eval_path, test_path, bsize, workers):
        super().__init__()
        self.prepare_data_per_node = False
        self.batch_size = bsize
        self.workers = workers
        self.train_path = train_path
        self.eval_path = eval_path
        self.test_path = test_path

    def setup(self, stage=None):
        self.train_dset = myloader.MyData(
            self.train_path
        )

        self.eval_dset = myloader.MyData(
            self.train_path
        )

        self.test_dset = myloader.MyData(
            self.train_path
        )

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dset,
            # collate_fn=collate_fn,
            num_workers=self.workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dset,
            # collate_fn=collate_fn,
            num_workers=self.workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_dset,
            # collate_fn=collate_fn,
            num_workers=self.workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )


def collate_fn(batch):
    # you can have your own collateral fn.
    print(batch)
    pass
