
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import os
import subprocess

from model_loader import load_models
from predict import predict_single
from utils.data_utils import load_input_data

models = load_models()

class RockburstApp:
    def __init__(self, master):
        self.master = master
        master.title("Rockburst Level Prediction App")
        master.geometry("800x600")

        self.bg_image = Image.open("assets/岩爆背景图1.jpg")
        self.bg_photo = ImageTk.PhotoImage(self.bg_image.resize((800, 600)))
        self.bg_label = tk.Label(master, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        tk.Label(master, text="Select Model:", bg="white").place(x=50, y=20)
        self.model_var = tk.StringVar(master)
        self.model_var.set("SVM")
        self.model_menu = tk.OptionMenu(master, self.model_var, *models.keys())
        self.model_menu.place(x=150, y=15)

        self.entries = []
        for i in range(7):
            tk.Label(master, text=f"D{i+1}:", bg="white").place(x=50, y=60 + i * 30)
            entry = tk.Entry(master)
            entry.place(x=150, y=60 + i * 30)
            self.entries.append(entry)

        self.predict_button = tk.Button(master, text="Predict", command=self.predict)
        self.predict_button.place(x=150, y=280)

        self.result_label = tk.Label(master, text="", bg="white", font=("Arial", 12))
        self.result_label.place(x=50, y=320)

        self.upload_button = tk.Button(master, text="Upload Excel & Train Models", command=self.upload_and_train)
        self.upload_button.place(x=50, y=360)

    def predict(self):
        try:
            values = [float(e.get()) for e in self.entries]
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numbers for D1 to D7.")
            return

        input_values = np.array(values).reshape(1, -1)
        model_name = self.model_var.get()
        model = models.get(model_name)

        if model is None:
            messagebox.showerror("Error", f"Model '{model_name}' not loaded.")
            return

        result = predict_single(model, input_values)
        text = f"Predicted Level: {result['label']}\n"
        if "probabilities" in result:
            for cls, prob in result["probabilities"].items():
                text += f"{cls}: {prob}\n"
        self.result_label.config(text=text)

    def upload_and_train(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return
        messagebox.showinfo("Training", "Model training has started. This may take several minutes.")
        try:
            subprocess.run(["python", "train_models.py", file_path], check=True)
            messagebox.showinfo("Done", "Models retrained and saved.")
        except Exception as e:
            messagebox.showerror("Training Failed", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = RockburstApp(root)
    root.mainloop()
