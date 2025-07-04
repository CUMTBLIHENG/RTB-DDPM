
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(path):
    df = pd.read_excel(path, sheet_name="Table 1")
    X = df[[f"D{i}" for i in range(1, 8)]].values
    y = df["Level"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    return df, X, X_scaled, y, y_encoded, scaler, le, class_names
