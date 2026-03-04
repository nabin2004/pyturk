"""Example: train XOR with pyturk MLP and Adam optimizer."""
from pyturk import Value
import pyturk.nn as nn
import pyturk.optim as optim
from pyturk.data import XORDataset


def train_xor(epochs=50, lr=0.05):
    ds = XORDataset()
    model = nn.MLP(2, [8, 8, 1], activation='tanh')
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = Value(0.0)
        correct = 0
        for x, y in zip(ds.X, ds.y):
            pred = model(x)
            loss = (pred - Value(y)) ** 2
            total_loss = total_loss + loss
            if (pred.data > 0 and y > 0) or (pred.data <= 0 and y <= 0):
                correct += 1

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | loss={total_loss.data:.4f} | acc={correct}/{len(ds)}')

    return model


if __name__ == '__main__':
    train_xor()
