import sys
import os
import pandas as pd
import joblib
from data_preprocessing import preprocess_data

# Insert the project root directory into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_model(model_path):
    """Load a saved model from disk."""
    return joblib.load(model_path)

def make_prediction(model, data):
    """Make a prediction using the trained model."""
    preprocessed_data = preprocess_data(data)

    # Ensure all expected columns are present
    expected_columns = [
        'absence_days', 'extracurricular_activities', 'weekly_self_study_hours',
        'math_score', 'history_score', 'physics_score', 'chemistry_score',
        'biology_score', 'english_score', 'geography_score',
        'career_aspiration_Artist', 'career_aspiration_Banker',
        'career_aspiration_Business Owner', 'career_aspiration_Construction Engineer',
        'career_aspiration_Designer', 'career_aspiration_Doctor',
        'career_aspiration_Game Developer', 'career_aspiration_Government Officer',
        'career_aspiration_Lawyer', 'career_aspiration_Missing',
        'career_aspiration_Real Estate Developer', 'career_aspiration_Scientist',
        'career_aspiration_Software Engineer', 'career_aspiration_Stock Investor',
        'career_aspiration_Teacher', 'career_aspiration_Writer',
        'gender_1', 'part_time_job_1'
    ]

    # Fill missing columns with zeros
    for col in expected_columns:
        if col not in preprocessed_data.columns:
            preprocessed_data[col] = 0

    preprocessed_data = preprocessed_data[expected_columns]

    # Make predictions
    prediction = model.predict(preprocessed_data)
    probability = model.predict_proba(preprocessed_data)
    return prediction, probability

def main():
    # Load the model
    model = load_model('neural_network_model.pkl')

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
    prediction, probability = make_prediction(model, new_data)
    print(f"Prediction: {prediction}, Probability: {probability}")

if __name__ == "__main__":
    main()