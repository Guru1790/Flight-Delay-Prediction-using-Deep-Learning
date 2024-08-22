import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib

# Load the pre-trained model and scaler
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

# Function to make a prediction
def predict_flight_delay(air_time, distance):
    # Prepare the user input
    user_input = np.array([[air_time, distance]])
    user_input_scaled = scaler.transform(user_input)
    
    # Make the prediction
    predictions = model.predict(user_input_scaled)
    
    predicted_delay_minutes = predictions[0][0]
    predicted_is_delay = predictions[0][1]
    
    # Determine if the flight is delayed
    if predicted_is_delay >= 0.5:
        return f"The flight is predicted to be delayed by {round(predicted_delay_minutes, 2)} minutes."
    else:
        return "The flight is predicted not to be delayed."

# Example usage
air_time = float(input("Enter Air Time in minutes: "))
distance = float(input("Enter Distance in miles: "))

result = predict_flight_delay(air_time, distance)
print(result)
