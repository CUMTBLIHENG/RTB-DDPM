# RTB-DDPM: Diffusion Model for Rockburst Tabular Data Generation

This repository implements **RTB-DDPM**, a novel diffusion-based generative model for synthesizing high-fidelity rockburst data, aiming to solve data imbalance and scarcity in geotechnical engineering.

## 🧠 Abstract

Rockburst is a common and high-risk hazard in deep underground engineering. While machine learning methods have shown promise for predicting rockburst intensity, they are hindered by limited and imbalanced data. RTB-DDPM addresses this by generating realistic synthetic data using a class-guided diffusion model.

Key features:
- Residual MLP architecture with temporal embedding
- Class-wise guided sampling
- Multiple synthesis strategies (PlanA, PlanB, PlanC_xx)
- Statistical evaluation using MMD, KLD, EMD, Cosine similarity
- Data visualization and loss curve tracking

## 📁 Project Structure

```
rockburst-ddpm/
├── data/                  # Raw data file (e.g., 300个案例.xlsx)
├── models/                # Diffusion model definitions
├── utils/                 # Evaluation metrics and preprocessing
├── generate.py           # Main training and generation script
├── requirements.txt      # Python dependencies
└── README.md             # Project overview
```

## 🚀 Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/your-username/rockburst-ddpm.git
cd rockburst-ddpm
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare your data

Put your Excel file in `data/300个案例.xlsx`.

### 4. Run the model

```bash
python generate.py
```

This script will:
- Train RTB-DDPM models per class
- Generate synthetic samples using multiple strategies
- Save visualizations and metrics to `ddpm_RT_generated_results_300/`

## 📊 Outputs

- Synthetic `.csv` and `.xlsx` data
- Loss curves per class
- KDE distribution comparisons
- Evaluation metrics (.csv summary)

## 📜 Citation

If you use this code, please cite:

> [Add your paper citation or link here]

## 📬 Contact

For questions or collaborations, please contact [your-email@example.com]
