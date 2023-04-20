from dataclasses import dataclass

import ml.api as ml
import torch
from omegaconf import MISSING
from torch import Tensor, nn
from torch.distributions.normal import Normal
from torch.nn import functional as F

from project.models.components import FeedForwardNet


@dataclass
class A2CContinuousModelConfig(ml.BaseModelConfig):
    state_dims: int = ml.conf_field(MISSING, help="The number of state dimensions")
    action_dims: int = ml.conf_field(MISSING, help="The number of action dimensions")
    hidden_dims: int = ml.conf_field(MISSING, help="The number of hidden dimensions")
    num_layers: int = ml.conf_field(MISSING, help="The number of hidden layers")
    activation: str = ml.conf_field("leaky_relu", help="The activation function to use")
    fixed_std: bool = ml.conf_field(False, help="Whether to use a fixed standard deviation")


@ml.register_model("a2c_continuous", A2CContinuousModelConfig)
class A2CContinuousModel(ml.BaseModel[A2CContinuousModelConfig]):
    fixed_std: Tensor | None

    def __init__(self, config: A2CContinuousModelConfig) -> None:
        super().__init__(config)

        self.policy_net = FeedForwardNet(
            config.state_dims,
            config.hidden_dims,
            config.action_dims if config.fixed_std else config.action_dims * 2,
            config.num_layers,
            act=ml.cast_activation_type(config.activation),
        )

        if config.fixed_std:
            self.register_parameter("fixed_std", nn.Parameter(torch.ones(1, config.action_dims)))
        else:
            self.register_parameter("fixed_std", None)

        self.value_net = FeedForwardNet(
            config.state_dims,
            config.hidden_dims,
            1,
            config.num_layers,
            act=ml.cast_activation_type(config.activation),
        )

    def forward_policy_net(self, state: Tensor) -> Normal:
        outputs = self.policy_net(state)

        if self.fixed_std is None:
            mean, std = outputs.tensor_split(2, dim=-1)
            std = F.softplus(std)
        else:
            mean, std = outputs, self.fixed_std

        mean, std = mean.clamp(-1e4, 1e4), std.clamp(1e-3, 1e4)
        return Normal(mean, std)

    def forward_value_net(self, state: Tensor) -> Tensor:
        return self.value_net(state)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        return self.policy_net(state), self.value_net(state)
