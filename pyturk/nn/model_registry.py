"""Pre-configured model registry for pyturk."""

from pyturk.nn.model import ModelBlock

MODEL_REGISTRY = {
    "Simple MLP": ModelBlock(
        name="Simple MLP", input_size=2, hidden_layers=[5], output_size=1,
    ),
    "Medium MLP": ModelBlock(
        name="Medium MLP", input_size=2, hidden_layers=[10, 5], output_size=1,
    ),
    "XOR MLP": ModelBlock(
        name="XOR MLP", input_size=2, hidden_layers=[4], output_size=1,
    ),
}
