from pyturk.nn.trainer import Trainer
from pyturk.nn.model_registry import MODEL_REGISTRY
from pyturk.data.moons import MoonsDataset

# Define simple loss function for demo
def loss_fn(model, X, y):
    preds = model(X)
    # simple mean squared error
    loss = ((preds - y)**2).sum()
    acc = ((preds > 0) == (y > 0)).float().mean().item()
    return loss, acc

# Dataset
dataset = MoonsDataset(n_samples=100)

# Model
model_block = MODEL_REGISTRY["Simple MLP"]

# Trainer
trainer = Trainer(model_block=model_block, dataset=dataset, loss_fn=loss_fn, steps=50)
history = trainer.train()
