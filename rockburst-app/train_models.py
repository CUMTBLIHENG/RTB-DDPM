
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def train_and_save_models(data_path, save_dir="models"):
    df = pd.read_excel(data_path)
    df = df.dropna()
    X = df[[f"D{i}" for i in range(1, 8)]].values
    y = df["Level"].values

    models_config = {
        "SVM": {
            "model": SVC(probability=True),
            "params": {
                "clf__C": [0.1, 1, 10],
                "clf__kernel": ["linear", "rbf"],
                "clf__gamma": ["scale", "auto"]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(),
            "params": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", tree_method="gpu_hist"),
            "params": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 6],
                "clf__learning_rate": [0.01, 0.1]
            }
        },
        "LightGBM": {
            "model": LGBMClassifier(device="gpu"),
            "params": {
                "clf__n_estimators": [100, 200],
                "clf__num_leaves": [31, 64],
                "clf__learning_rate": [0.01, 0.1]
            }
        },
        "CatBoost": {
            "model": CatBoostClassifier(verbose=0, task_type="GPU"),
            "params": {
                "clf__depth": [4, 6],
                "clf__learning_rate": [0.01, 0.1],
                "clf__iterations": [100, 200]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "clf__n_neighbors": [3, 5, 7],
                "clf__weights": ["uniform", "distance"]
            }
        }
    }

    for name, cfg in models_config.items():
        print(f"🔍 Training {name}...")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", cfg["model"])
        ])
        grid = GridSearchCV(pipe, cfg["params"], cv=3, scoring="accuracy", n_jobs=-1)
        grid.fit(X, y)
        best_model = grid.best_estimator_
        model_dir = os.path.join(save_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{name}_best_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"✅ Saved {name} model to {model_path}")
