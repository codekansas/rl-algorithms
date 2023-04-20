from dataclasses import dataclass

import ml.api as ml
from omegaconf import MISSING
from torch import Tensor
from torch.distributions.categorical import Categorical

from project.models.components import FeedForwardNet


@dataclass
class A2CDiscreteModelConfig(ml.BaseModelConfig):
    state_dims: int = ml.conf_field(MISSING, help="The number of state dimensions")
    action_dims: int = ml.conf_field(MISSING, help="The number of action dimensions")
    hidden_dims: int = ml.conf_field(MISSING, help="The number of hidden dimensions")
    num_layers: int = ml.conf_field(MISSING, help="The number of hidden layers")
    activation: str = ml.conf_field("leaky_relu", help="The activation function to use")


@ml.register_model("a2c_discrete", A2CDiscreteModelConfig)
class A2CDiscreteModel(ml.BaseModel[A2CDiscreteModelConfig]):
    fixed_std: Tensor | None

    def __init__(self, config: A2CDiscreteModelConfig) -> None:
        super().__init__(config)

        self.policy_net = FeedForwardNet(
            config.state_dims,
            config.hidden_dims,
            config.action_dims,
            config.num_layers,
            act=ml.cast_activation_type(config.activation),
        )

        self.value_net = FeedForwardNet(
            config.state_dims,
            config.hidden_dims,
            1,
            config.num_layers,
            act=ml.cast_activation_type(config.activation),
        )

    def forward_policy_net(self, state: Tensor) -> Categorical:
        outputs = self.policy_net(state)
        return Categorical(logits=outputs)

    def forward_value_net(self, state: Tensor) -> Tensor:
        return self.value_net(state)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        return self.policy_net(state), self.value_net(state)
