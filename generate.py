
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from models.ddpm import ImprovedDDPM
from utils.metrics import compute_mmd, compute_kld, compute_emd, compute_cosine_sim

# 参数设定
SAVE_DIR = "ddpm_RT_generated_results"
os.makedirs(SAVE_DIR, exist_ok=True)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

EPOCHS = 3000
BATCH_SIZE = 64
LATENT_DIM = 7
LEARNING_RATE = 2e-4
NOISE_STEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

# 数据读取与处理
df = pd.read_excel("data/your_data.xlsx", sheet_name="Table 1")
X = df[[f"D{i}" for i in range(1, 8)]].values
y = df["Level"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
NUM_CLASSES = len(np.unique(y_encoded))
class_names = le.classes_

original_counts = {i: sum(y_encoded == i) for i in range(NUM_CLASSES)}
mean_class_count = int(np.mean(list(original_counts.values())))
max_class_count = max(original_counts.values())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DDPM 参数
betas = torch.linspace(BETA_START, BETA_END, NOISE_STEPS).to(device)
alphas = 1. - betas
alpha_hat = torch.cumprod(alphas, dim=0)

# 添加噪声
def noise_sample(x, t):
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None]
    eps = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

# 生成策略定义
strategies = {
    'PlanA': lambda c: max_class_count,
    'PlanB': lambda c: max(mean_class_count, original_counts[c])
}
for mult in range(1, 11):
    strategies[f'PlanC_{mult}x'] = lambda c, m=mult: mean_class_count * m + original_counts[c]

# 主训练与生成循环
for strategy, count_fn in strategies.items():
    print(f"\n--- Running {strategy} ---")
    results = []
    all_synth_df = []

    for class_id in range(NUM_CLASSES):
        class_name = class_names[class_id]
        X_train = X_scaled[y_encoded == class_id]
        n_samples = len(X_train)
        n_generate = count_fn(class_id)

        model = ImprovedDDPM(input_dim=X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.HuberLoss()
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        loss_curve = []

        for epoch in range(EPOCHS):
            idx = torch.randint(0, n_samples, (BATCH_SIZE,))
            x = X_tensor[idx]
            t = torch.randint(0, NOISE_STEPS, (BATCH_SIZE,), device=device)
            x_t, noise = noise_sample(x, t)
            predicted = model(x_t, t)
            loss = loss_fn(predicted, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())

        # 保存损失曲线
        plt.figure()
        plt.plot(loss_curve)
        plt.title(f"Loss Curve - {strategy} - {class_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"{strategy}_class_{class_name}_loss.png"))
        plt.close()

        # 合成样本
        model.eval()
        with torch.no_grad():
            x = torch.randn(n_generate, LATENT_DIM).to(device)
            for t_step in reversed(range(NOISE_STEPS)):
                t = torch.full((n_generate,), t_step, device=device, dtype=torch.long)
                z = torch.randn_like(x) if t_step > 0 else 0
                predicted_noise = model(x, t)
                alpha = alphas[t][:, None]
                alpha_hat_t = alpha_hat[t][:, None]
                beta = betas[t][:, None]
                x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat_t) * predicted_noise) + torch.sqrt(beta) * z

            generated = x.cpu().numpy()
            generated_rescaled = scaler.inverse_transform(generated)

        generated_rescaled = np.clip(generated_rescaled, a_min=0, a_max=None)
        df_gen = pd.DataFrame(generated_rescaled, columns=[f"D{i}" for i in range(1, 8)])
        df_gen["Level"] = class_name
        df_gen.to_csv(os.path.join(SAVE_DIR, f"{strategy}_class_{class_name}_samples.csv"), index=False)
        all_synth_df.append(df_gen)

        # 评估指标
        mmd_val = compute_mmd(X_train, generated)
        kld_val = compute_kld(X_train, generated)
        emd_val = compute_emd(X_train, generated)
        cosine_val = compute_cosine_sim(X_train, generated)

        results.append({
            "strategy": strategy,
            "class": class_name,
            "samples_generated": n_generate,
            "MMD": mmd_val,
            "KLD": kld_val,
            "EMD": emd_val,
            "CosineSim": cosine_val
        })

    # 合并所有类别样本
    merged_df = pd.concat(all_synth_df, ignore_index=True)
    merged_df.to_excel(os.path.join(SAVE_DIR, f"{strategy}_all_samples.xlsx"), index=False)

    # KDE 图对比
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, col in enumerate([f"D{i}" for i in range(1, 8)]):
        try:
            real_col = pd.to_numeric(df[col], errors='coerce').dropna()
            synth_col = pd.to_numeric(merged_df[col], errors='coerce').dropna()
            sns.kdeplot(real_col, ax=axes[i], label='Original', fill=True, alpha=0.5)
            sns.kdeplot(synth_col, ax=axes[i], label='Generated', fill=True, alpha=0.5)
            axes[i].set_title(f'Density: {col}')
            axes[i].legend()
        except Exception as e:
            axes[i].set_title(f"Error: {col}")
            print(f"Plot error on {col}: {e}")
    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{strategy}_density_comparison.png"))
    plt.close()

    # 保存评估结果
    results_df = pd.DataFrame(results)
    avg_metrics = results_df[['MMD', 'KLD', 'EMD', 'CosineSim']].mean().to_dict()
    avg_metrics.update({
        'strategy': f"{strategy}-Average",
        'class': "Average",
        'samples_generated': results_df['samples_generated'].sum()
    })
    results_df = pd.concat([results_df, pd.DataFrame([avg_metrics])], ignore_index=True)
    results_df.to_csv(os.path.join(SAVE_DIR, f"{strategy}_metrics.csv"), index=False)

    summary_path = os.path.join(SAVE_DIR, "all_strategy_averages.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        summary_df = pd.concat([summary_df, pd.DataFrame([avg_metrics])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([avg_metrics])
    summary_df.to_csv(summary_path, index=False)

    print(f"\n[Strategy: {strategy}] 平均指标:")
    for key, val in avg_metrics.items():
        if key not in ["strategy", "class"]:
            print(f"  {key}: {val:.4f}")

print("✅ RTB-DDPM data generation completed.")
