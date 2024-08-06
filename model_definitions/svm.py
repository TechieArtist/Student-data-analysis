from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from source.data_preprocessing import load_data, preprocess_data  

def train_svm(X_train, y_train):
    """Train an SVM model."""
    model = SVC(probability=True)  
    model.fit(X_train, y_train)
    return model

def evaluate_svm(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, conf_matrix

def main():
    # Load and preprocess the data
    df = load_data('data/processed/processed_student_scores_general.csv')
    df = preprocess_data(df)

    # Split the data into training and testing sets
    X = df.drop('Passed', axis=1)  
    y = df['Passed']  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the SVM model
    svm_model = train_svm(X_train, y_train)

    # Evaluate the model
    accuracy, report, conf_matrix = evaluate_svm(svm_model, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Save the model
    joblib.dump(svm_model, 'svm_model_with_proba.pkl')  # Save the trained model

if __name__ == "__main__":
    main()
