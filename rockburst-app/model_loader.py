
import joblib
from config import MODEL_PATHS

def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        try:
            models[name] = joblib.load(path)
            print(f"✅ Loaded model: {name}")
        except Exception as e:
            models[name] = None
            print(f"❌ Failed to load model: {name} | Error: {e}")
    return models
