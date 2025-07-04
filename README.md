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
├── data/                  # Raw data file 
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

# 🧠 Rockburst Level Prediction App

This is an intelligent graphical application for rockburst level prediction using multiple machine learning models.  
It supports real-time single-point prediction, Excel-based batch training, and model evaluation — all wrapped in a simple GUI.

## 📌 Features

- ✅ GUI-based rockburst intensity prediction  
- 🧪 Support for 6 ML models: SVM, Random Forest, XGBoost, LightGBM, CatBoost, KNN  
- 📈 Hyperparameter tuning with GridSearchCV  
- 📊 ROC curve generation and evaluation (F1, Accuracy)  
- 🗂️ Supports Excel (.xlsx) for data input  
- 🧠 Real-time classification with prediction probabilities  

## 🖼️ GUI Screenshot (Placeholder)

> Replace with an actual image:

## 📁 Project Structure

rockburst-app/
├── 🧠 app_gui.py # Main GUI interface (Tkinter-based)
├── ⚙️ config.py # Label mappings, model paths, CV settings
├── 📦 model_loader.py # Load pre-trained .pkl models
├── 🤖 predict.py # Single-sample prediction logic
├── 🏋️ train_models.py # GridSearchCV training for 6 models
├── 📄 requirements.txt # Python package requirements
├── 📝 README.md # Project documentation
│
├── 🔧 utils/ # Utility scripts
│ └── 🧾 data_utils.py # Excel reader and preprocessor
│
├── 📁 models/ # Saved model folders
│ ├── SVM/
│ │ └── SVM_best_model.pkl
│ ├── RandomForest/
│ ├── XGBoost/
│ ├── LightGBM/
│ ├── CatBoost/
│ └── KNN/
│
└── 🖼️ assets/ # GUI assets
└── 岩爆背景图1.jpg # Background image

