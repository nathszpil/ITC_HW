import pickle
import pandas as pd
import numpy as np
from flask import Flask, request

app = Flask(__name__)


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def test_model(model, X_test_path, preds_path):
    X_test = pd.read_csv(X_test_path)
    y_true = np.loadtxt(preds_path, delimiter=',')
    y_pred = model.predict(X_test)
    if np.array_equal(y_pred, y_true):
        return True
    else:
        return False


def predict_churn(features):
    input_data = pd.DataFrame([features])
    prediction = model.predict(input_data)
    return str(prediction[0])


model_path = 'churn_model.pkl'
X_test_path = 'X_test.csv'
preds_path = 'preds.csv'

model = load_model(model_path)
test_passed = test_model(model, X_test_path, preds_path)

if test_passed:
    print("Model test passed. Starting Flask server...")
else:
    print("Model test failed")


@app.route('/predict_churn', methods=['GET'])
def predict():
    feature1 = float(request.args.get('is_male'))
    feature2 = float(request.args.get('num_inters'))
    feature3 = float(request.args.get('late_on_payment'))
    feature4 = float(request.args.get('age'))
    feature5 = float(request.args.get('years_in_contract'))

    prediction = predict_churn({'is_male': feature1, 'num_inters': feature2, 'late_on_payment': feature3,
                                'age': feature4, 'years_in_contract': feature5})

    return prediction


if __name__ == '__main__':
    app.run(debug=True, port=5001)
