#!/usr/bin/env python3
# -------------------------------------------------------
# Custom net builder
# Author: Neeraj Jain
# -------------------------------------------------------

from typing import List

from reagent.core import types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import NormalizationData, param_hash
from reagent.models.base import ModelBase
from reagent.models.mobnet import MobModel
from reagent.net_builder.discrete_dqn_net_builder import DiscreteDQNNetBuilder
import torchvision.models as models
import torch


@dataclass
class MobNet(DiscreteDQNNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    dropout_ratio: float = 0.0
    use_batch_norm: bool = False

    def __post_init_post_parse__(self) -> None:
        super().__init__()
        assert len(self.sizes) == len(self.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{self.sizes}, {self.activations}"
        )

    def build_q_network(
        self,
        state_feature_config: rlt.ModelFeatureConfig,
        cfg,
    ) -> ModelBase:

        name = cfg.MODEL.name
        channels = cfg.MODEL.channels
        classes = cfg.MODEL.classes
        model = MobModel(name, channels, classes)

        return model
