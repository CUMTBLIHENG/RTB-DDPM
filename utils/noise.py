
import torch
from config import *

def get_noise_schedule():
    betas = torch.linspace(BETA_START, BETA_END, NOISE_STEPS).to(device)
    alphas = 1. - betas
    alpha_hat = torch.cumprod(alphas, dim=0)
    return alphas, alpha_hat, betas

def noise_sample(x, t, alpha_hat):
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None]
    eps = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps
