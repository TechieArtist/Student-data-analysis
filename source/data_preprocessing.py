from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler




def load_data(filepath):
    """Loads data from a CSV file."""
    return pd.read_csv(filepath)

expected_columns = [
    'absence_days', 'weekly_self_study_hours', 'math_score', 'history_score',
    'physics_score', 'chemistry_score', 'biology_score', 'english_score',
    'geography_score', 'career_aspiration_Artist', 'career_aspiration_Banker', 
    'career_aspiration_Business Owner', 'career_aspiration_Construction Engineer',
    'career_aspiration_Designer', 'career_aspiration_Doctor', 'career_aspiration_Game Developer',
    'career_aspiration_Government Officer', 'career_aspiration_Lawyer', 'career_aspiration_Missing',
    'career_aspiration_Real Estate Developer', 'career_aspiration_Scientist', 
    'career_aspiration_Software Engineer', 'career_aspiration_Stock Investor', 
    'career_aspiration_Teacher', 'career_aspiration_Writer', 'gender_1', 'part_time_job_1'
]

def preprocess_data(df, scaler=None):
    """Preprocesses the data by handling missing values, scaling features, and encoding labels."""
    
    # Handling missing values for numerical columns
    numerical_columns = [
        'math_score', 'history_score', 'physics_score', 'chemistry_score',
        'biology_score', 'english_score', 'geography_score'
    ]
    df[numerical_columns] = df[numerical_columns].apply(lambda x: x.fillna(x.mean()), axis=0)

    # Handle 'Unknown' and missing values in career_aspiration column
    df['career_aspiration'] = df['career_aspiration'].replace('Unknown', 'Missing')

    # One-Hot Encode categorical variables with consistent column names
    df = pd.get_dummies(df, columns=['career_aspiration', 'gender', 'part_time_job'], drop_first=True)

    # Manually adjust column names for consistency if necessary
    if 'gender_Male' in df.columns:
        df.rename(columns={'gender_Male': 'gender_1'}, inplace=True)
    if 'part_time_job_True' in df.columns:
        df.rename(columns={'part_time_job_True': 'part_time_job_1'}, inplace=True)

    # Feature Scaling
    if scaler is None:
        scaler = MinMaxScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    else:
        df[numerical_columns] = scaler.transform(df[numerical_columns])

    # Create a 'Passed' column based on a condition (used in training, but not in prediction)
    if 'Passed' not in df.columns:
        df['Passed'] = (df[numerical_columns].mean(axis=1) > 0.6).astype(int)

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Set missing columns to a default value

    # Ensure the order of columns matches expected_columns
    df = df[expected_columns + (['Passed'] if 'Passed' in df.columns else [])]

    return df, scaler


def split_data(df):
    """Splits the data into training and testing sets."""
    # Columns to drop if they exist in the dataframe
    columns_to_drop = ['id', 'first_name', 'last_name', 'email']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    # Drop non-numeric or irrelevant columns
    X = df.drop(existing_columns_to_drop, axis=1)

    # Handle the case where 'Passed' may not exist
    if 'Passed' in df.columns:
        y = df['Passed']
        X = X.drop(['Passed'], axis=1)
    else:
        y = None

    # Return X, y if y exists, otherwise, return X only
    if y is not None:
        return train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        return X, None, None, None
