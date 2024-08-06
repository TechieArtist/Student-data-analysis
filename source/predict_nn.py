import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import joblib
import torch
import pandas as pd
from data_preprocessing import preprocess_data
from model_definitions.neural_network import SimpleNNWithAttention

def load_model(model_path):
    """Load a saved PyTorch model from disk."""
    print(f"Loading model from {model_path}")
    model = SimpleNNWithAttention(input_size=28, hidden_size=10, output_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded and set to evaluation mode")
    return model

def make_prediction(model, data, scaler):
    """Make a prediction using the trained model."""
    print("Preprocessing data...")
    # Use the scaler for transformation
    preprocessed_data, _ = preprocess_data(data, scaler=scaler)
    print(f"Preprocessed Data Columns: {preprocessed_data.columns.tolist()}")

    # Select only numerical columns after preprocessing
    preprocessed_data = preprocessed_data.select_dtypes(include=[np.number])
    print(f"Data after selecting numeric types: {preprocessed_data}")

    # Convert preprocessed data to numpy array and then to tensor
    data_tensor = torch.tensor(preprocessed_data.values, dtype=torch.float32)
    print(f"Data tensor shape: {data_tensor.shape}")

    # Ensure the input size matches the model's expected input size
    expected_input_size = model.fc1.in_features
    actual_input_size = data_tensor.shape[1]
    assert actual_input_size == expected_input_size, f"Mismatch in input feature size. Expected: {expected_input_size}, Got: {actual_input_size}"

    # Make prediction
    with torch.no_grad():
        outputs = model(data_tensor)
        prediction = (outputs >= 0.5).float()  # Threshold at 0.5 for binary classification
        probability = outputs

    return prediction.numpy(), probability.numpy()

def main():
    # Load the model
    model_path = 'models/neural_network_model.pt'
    print(f"Model path: {model_path}")
    model = load_model(model_path)

    # Load the scaler
    scaler_path = 'models/scaler.pkl'
    scaler = joblib.load(scaler_path)
    print("Scaler loaded")

    # New data for prediction
    new_data = pd.DataFrame({
        'id': [1], 'first_name': ['John'], 'last_name': ['Doe'], 'email': ['john.doe@example.com'],
        'absence_days': [3], 'extracurricular_activities': [False], 'weekly_self_study_hours': [10],
        'math_score': [0.75], 'history_score': [0.65], 'physics_score': [0.55], 'chemistry_score': [0.60],
        'biology_score': [0.70], 'english_score': [0.80], 'geography_score': [0.65],
        'career_aspiration': ['Software Engineer'], 'gender': ['Male'], 'part_time_job': [False]
    })

    # Debug print to verify columns in new data
    print(f"New Data Columns: {new_data.columns.tolist()}")

    # Make predictions
    prediction, probability = make_prediction(model, new_data, scaler)
    print(f"Prediction: {prediction}, Probability: {probability}")

if __name__ == "__main__":
    main()


