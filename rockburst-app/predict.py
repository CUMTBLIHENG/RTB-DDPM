
from config import LABEL_MAP

def predict_single(model, input_values):
    y_pred = model.predict(input_values)[0]
    result = {
        "label": LABEL_MAP.get(y_pred, str(y_pred))
    }
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(input_values)[0]
        result["probabilities"] = {LABEL_MAP[i]: round(p, 3) for i, p in enumerate(probas)}
    return result
