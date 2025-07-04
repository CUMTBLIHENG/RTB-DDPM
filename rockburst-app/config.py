
LABEL_MAP = {
    0: "Intense",
    1: "Moderate",
    2: "Light",
    3: "Safe"
}

MODEL_PATHS = {
    "SVM": "models/SVM/SVM_best_model.pkl",
    "XGBoost": "models/XGBoost/XGBoost_best_model.pkl",
    "RandomForest": "models/RandomForest/RandomForest_best_model.pkl",
    "KNN": "models/KNN/KNN_best_model.pkl",
    "MLP": "models/MLP/MLP_best_model.pkl"
}

CV_SPLITS = 5
