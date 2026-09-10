import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# class my_class(object):
#     pass

# Sample dataset
def build_and_save_model(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop("target", axis=1)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)




