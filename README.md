# RTB-DDPM: Diffusion Model for Rockburst Tabular Data Generation

This repository implements **RTB-DDPM**, a novel diffusion-based generative model for synthesizing high-fidelity rockburst data, aiming to solve data imbalance and scarcity in geotechnical engineering.

## 🧠 Abstract

Rockburst is a common and high-risk hazard in deep underground engineering. While machine learning methods have shown promise for predicting rockburst intensity, they are hindered by limited and imbalanced data. RTB-DDPM addresses this by generating realistic synthetic data using a class-guided diffusion model.

Key features:
- Residual MLP architecture with temporal embedding
- Class-wise guided sampling
- Multiple synthesis strategies (PlanA, PlanB, PlanC_xx)
- Data visualization and loss curve tracking

## 📁 Project Structure

```
rockburst-ddpm/
├── data/                  # Raw data file (e.g., 300个案例.xlsx)
├── models/                # Diffusion model definitions
├── utils/                 # preprocessing
├── generate.py           # Main training and generation script
├── requirements.txt      # Python dependencies
└── README.md             # Project overview
```



## 📊 Outputs

- Synthetic `.csv` and `.xlsx` data
- Loss curves per class
- Evaluation metrics (.csv summary)

## 📜 Citation

If you use this code, please cite:


