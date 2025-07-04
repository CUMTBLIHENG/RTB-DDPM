
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from config import CV_SPLITS

def evaluate_model(model, X, y):
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy").mean()
    f1 = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted").mean()
    return {"accuracy": acc, "f1_score": f1}
