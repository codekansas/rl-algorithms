import ml.api as ml
from torch import Tensor, nn


class FeedForwardNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act: ml.ActivationType = "leaky_relu",
        norm: ml.NormType = "layer_affine",
    ) -> None:
        super().__init__()

        # Saves the model parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.act = act

        # Instantiates the model layers.
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            self.layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                self.layers.append(ml.get_norm_linear(norm, dim=out_dim))
                self.layers.append(ml.get_activation(act))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Makes the last layer very small.
        last_layer = self.layers[-1]
        assert isinstance(last_layer, nn.Linear)
        nn.init.xavier_uniform_(last_layer.weight, gain=0.01)
        nn.init.zeros_(last_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
