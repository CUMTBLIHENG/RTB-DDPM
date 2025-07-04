
import torch
import numpy as np
from config import *
from utils.noise import get_noise_schedule

def generate_samples(model, n_generate, latent_dim, scaler):
    alphas, alpha_hat, betas = get_noise_schedule()
    model.eval()
    with torch.no_grad():
        x = torch.randn(n_generate, latent_dim).to(device)
        for t_step in reversed(range(len(betas))):
            t = torch.full((n_generate,), t_step, device=device, dtype=torch.long)
            z = torch.randn_like(x) if t_step > 0 else 0
            predicted_noise = model(x, t)
            alpha = alphas[t][:, None]
            alpha_hat_t = alpha_hat[t][:, None]
            beta = betas[t][:, None]
            x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat_t) * predicted_noise) + torch.sqrt(beta) * z

        generated = x.cpu().numpy()
        generated_rescaled = scaler.inverse_transform(generated)
        return np.clip(generated_rescaled, a_min=0, a_max=None)
