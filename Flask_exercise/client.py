import pandas as pd
import numpy as np
import requests

X_test = pd.read_csv('X_test.csv')
y_true = np.loadtxt('preds.csv', delimiter=',')


def get_prediction(features):
    url = 'http://localhost:5001/predict_churn'
    response = requests.get(url, params=features)
    return response.text


for i in range(5):
    features = {key: value for key, value in X_test.iloc[i].to_dict().items()}
    prediction = get_prediction(features)
    print(f"Observation {i + 1}: Predicted churn - {prediction}, Actual churn - {int(y_true[i])}")
