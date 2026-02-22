from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Smart EV Charging Demand Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_data = pd.DataFrame({
        'day_of_week': [data['day_of_week']],
        'time_hour': [data['time_hour']],
        'charging_duration_minutes': [data['charging_duration_minutes']],
        'slots_available': [data['slots_available']]
    })

    prediction = model.predict(input_data)

    return jsonify({
        "predicted_demand": round(float(prediction[0]))
    })

if __name__ == "__main__":
    app.run(debug=True)
