
import torch
import numpy as np
import os

# 训练参数
SEED = 42
EPOCHS = 3000
BATCH_SIZE = 64
LATENT_DIM = 7
LEARNING_RATE = 2e-4
NOISE_STEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

SAVE_DIR = "ddpm_RT_generated_results_300"
os.makedirs(SAVE_DIR, exist_ok=True)

# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
