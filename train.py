
import torch
import torch.nn as nn
from config import *
from utils.noise import noise_sample

def train_model(X_tensor, model, optimizer, loss_fn, alpha_hat, noise_steps):
    n_samples = X_tensor.shape[0]
    loss_curve = []
    for epoch in range(EPOCHS):
        idx = torch.randint(0, n_samples, (BATCH_SIZE,))
        x = X_tensor[idx]
        t = torch.randint(0, noise_steps, (BATCH_SIZE,), device=device)
        x_t, noise = noise_sample(x, t, alpha_hat)
        predicted = model(x_t, t)
        loss = loss_fn(predicted, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_curve.append(loss.item())
    return model, loss_curve
