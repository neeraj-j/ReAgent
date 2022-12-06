#!/usr/bin/env python3
# ------------------------------------------------------------
# Custom model builder 
# Author: Neeraj Jain
# ----------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn
from reagent.core import types as rlt
import torchvision.models as models

INVALID_ACTION_CONSTANT: float = -1e10


class MobModel(nn.Module):
    def __init__(
        self,
        name,
        channels,
        classes,
    ):
        super().__init__()

        model = eval("models." + name)(weights=None)
        # modify model
        if name == "mobilenet_v3_large":
            # first layer
            model.features[0] = torch.nn.Conv2d(
                channels,
                16,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            )
            # last layer
            model.classifier[3] = torch.nn.Linear(
                in_features=1280, out_features=classes, bias=True
            )
        elif name == "efficientnet_b3":
            # first layer
            model.features[0][0] = torch.nn.Conv2d(
                channels,
                40,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            )
            # last layer
            model.classifier[1] = torch.nn.Linear(
                in_features=1536, out_features=classes, bias=True
            )
        elif name == "efficientnet_b2":
            # first layer
            model.features[0][0] = torch.nn.Conv2d(
                channels,
                32,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            )
            # last layer
            model.classifier[1] = torch.nn.Linear(
                in_features=1408, out_features=classes, bias=True
            )

        self.model = model

    def forward(
        self,
        state: rlt.FeatureData,
        possible_actions_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = state.float_features
        x = self.model(x)

        if possible_actions_mask is not None:
            # subtract huge value from impossible actions to
            # force their probabilities to 0
            x = x + (1 - possible_actions_mask.float()) * INVALID_ACTION_CONSTANT
        return x
