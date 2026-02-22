import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv("data/ev_charging_data.csv")

data['day_of_week'] = data['day_of_week'].astype('category').cat.codes
data['time_hour'] = pd.to_datetime(data['time']).dt.hour

X = data[['day_of_week', 'time_hour', 'charging_duration_minutes', 'slots_available']]
y = data['demand']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Model Performance:")
print("MAE:", mae)
print("RMSE:", rmse)

os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully in models/model.pkl")
