import pickle
import pandas as pd

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

sample_data = pd.DataFrame({
    'day_of_week': [2],
    'time_hour': [18],
    'charging_duration_minutes': [60],
    'slots_available': [1]
})

prediction = model.predict(sample_data)

print("Predicted EV Charging Demand:", round(prediction[0]))
