# ---------------------------------------------------------------------
# loads the data
# Author Neeraj Jain
# --------------------------------------------------------------------
import os, sys
import numpy as np
import math
import copy

import torch
from glob import glob
import pandas as pd
import random
from torch.utils.data import Dataset
import torch.nn.functional as F

random.seed(0)


class MyData(Dataset):
    def __init__(self, root):
        self.root = root
        self.db = self.load_db(
            root
        )

    def __getitem__(self, idx):
        # create dataframe from your db
        # following is a dummy dataframe generation
        state = torch.zeros((64, 128))
        next_state = torch.zeros((64, 128))
        action = torch.tensor(1)
        next_action = torch.tensor(0)

        # expand channel dim
        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)
        dic = {}
        dic["state_features"] = state
        dic["next_state_features"] = next_state
        dic["action"] = F.one_hot(action, num_classes=2)
        dic["next_action"] = F.one_hot(next_action, num_classes=2)
        dic["reward"] = torch.tensor([1.0])
        dic["time_diff"] = torch.tensor([1.0])
        dic["step"] = torch.tensor([idx])
        dic["not_terminal"] = torch.tensor([1.0])
        dic["possible_actions_mask"] = torch.tensor([1, 1])
        dic["possible_next_actions_mask"] = torch.tensor([1, 1])
        dic['action_probability'] = torch.tensor([0.5])
        return dic

    def load_db(self, path):
        # pocess and laod your data
        db = [10] * 10000
        return db

    def __len__(
        self,
    ):
        return len(self.db)


