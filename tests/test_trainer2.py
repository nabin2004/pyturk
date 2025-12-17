from pyturk.nn.trainer import Trainer
from pyturk.nn.model_registry import MODEL_REGISTRY
from pyturk.data.moons import MoonsDataset

# Dataset
dataset = MoonsDataset(n_samples=100)

# Model
model_block = MODEL_REGISTRY["Simple MLP"]

# Loss function
def mse_loss(model, X, y):
    preds = model(X)
    loss = ((preds - y)**2).sum()
    acc = ((preds > 0) == (y > 0)).float().mean().item()
    return loss, acc

# Trainer
trainer = Trainer(model_block=model_block, dataset=dataset, loss_fn=mse_loss, steps=50)
history = trainer.train()
