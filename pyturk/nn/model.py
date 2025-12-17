from pyturk.nn.mlp import MLP

class ModelBlock:
    """
    A block that wraps a neural network with configurable parameters.
    """

    def __init__(self, name, input_size=2, hidden_layers=[5], output_size=1, activation='tanh'):
        self.name = name
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Create a fresh MLP with current parameters."""
        return MLP(nin=self.input_size, nouts=self.hidden_layers + [self.output_size])

    def reset(self, input_size=None, hidden_layers=None, output_size=None, activation=None):
        """Reset model with new parameters (user tweaks)."""
        if input_size is not None:
            self.input_size = input_size
        if hidden_layers is not None:
            self.hidden_layers = hidden_layers
        if output_size is not None:
            self.output_size = output_size
        if activation is not None:
            self.activation = activation

        self.model = self._initialize_model()
