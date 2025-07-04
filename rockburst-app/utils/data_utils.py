
import pandas as pd
import numpy as np

def load_input_data(file_path):
    df = pd.read_excel(file_path)
    df = df.dropna()
    X = df[[f"D{i}" for i in range(1, 8)]].values
    y = df["Level"].values
    return X, y
