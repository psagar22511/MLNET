# class my_class(object):
#     pass

import os
import joblib
import numpy as np

# Import the function from your train.py file
from .train import build_and_save_model 

model_path = r"C:\SagarPatel\Practice\ML.NET\AI.Microservice.Platform\ml-service\app\schemas\model.pkl"
data_path = r"C:\SagarPatel\Practice\ML.NET\AI.Microservice.Platform\ml-service\app\schemas\data.csv"

# Automatically trigger train.py function if file is missing
if not os.path.exists(model_path):
    print("⚠️ Model missing. Triggering train.py function...")
    build_and_save_model(data_path, model_path)
else:
    print("✅ Model found. Loading the model...")

model = joblib.load(model_path)

def predict(data):
    input_array = np.array(data).reshape(1, -1)
    return model.predict(input_array)[0]



