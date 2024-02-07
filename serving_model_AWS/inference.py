import pickle
import pandas as pd
from flask import Flask, request


app = Flask(__name__)


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_churn(features):
    input_data = pd.DataFrame([features])
    prediction = model.predict(input_data)
    return str(prediction[0])


model_path = 'churn_model.pkl'
model = load_model(model_path)


@app.route('/predict_churn', methods=['GET'])
def predict():
    is_male = float(request.args.get('is_male'))
    num_inters = float(request.args.get('num_inters'))
    late_on_payement = float(request.args.get('late_on_payment'))
    age = float(request.args.get('age'))
    years_in_contract = float(request.args.get('years_in_contract'))

    prediction = predict_churn({'is_male': is_male, 'num_inters': num_inters, 'late_on_payment':late_on_payement,
                                'age':age, 'years_in_contract':years_in_contract})

    return prediction


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
