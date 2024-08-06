import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from sklearn.model_selection import train_test_split, cross_val_score
from model_definitions.svm import train_svm, evaluate_svm
from model_definitions.logistic_regression import train_logistic_regression, evaluate_logistic_regression
from model_definitions.decision_tree import train_decision_tree, evaluate_decision_tree
from model_definitions.neural_network import train_neural_network, evaluate_neural_network
from source.data_preprocessing import load_data, preprocess_data, split_data
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import joblib

def train_and_evaluate(model_type, X_train, y_train, X_test, y_test):
    if model_type == 'svm':
        model = train_svm(X_train, y_train)
        accuracy, report, conf_matrix = evaluate_svm(model, X_test, y_test)
        joblib.dump(model, 'models/svm_model.pkl')
    elif model_type == 'logistic_regression':
        model = train_logistic_regression(X_train, y_train)
        accuracy, report, conf_matrix = evaluate_logistic_regression(model, X_test, y_test)
        joblib.dump(model, 'models/logistic_regression_model.pkl')
    elif model_type == 'decision_tree':
        model = train_decision_tree(X_train, y_train)
        accuracy, report, conf_matrix = evaluate_decision_tree(model, X_test, y_test)
        joblib.dump(model, 'models/decision_tree_model.pkl')
    elif model_type == 'neural_network':
        input_size = X_train.shape[1]
        hidden_size = 10
        output_size = 1
        model = train_neural_network(X_train, y_train, input_size, hidden_size, output_size)
        accuracy, report, conf_matrix = evaluate_neural_network(model, X_test, y_test)
        torch.save(model.state_dict(), 'models/neural_network_model.pt')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return accuracy, report, conf_matrix

def main():
    # Create the models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Load and preprocess data
    df = load_data('data/processed/processed_student_scores_general.csv')
    df, scaler = preprocess_data(df)  # Unpack both dataframe and scaler

    # Inspect the columns to confirm the correct column names
    print("Columns in the DataFrame:", df.columns.tolist())

    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    if y_train is None or y_test is None:
        print("No target variable ('Passed') found in the dataset. Cannot proceed with training.")
        return

    # Use the training dataset for cross-validation
    X, y = X_train, y_train

    # Define models for cross-validation
    svm_model = SVC()
    log_reg_model = LogisticRegression(max_iter=1000, C=1.0, penalty='l2')
    decision_tree_model = DecisionTreeClassifier()

    # Cross-Validation for SVM
    svm_scores = cross_val_score(svm_model, X, y, cv=5)
    print(f"SVM Cross-Validation Scores: {svm_scores}")
    print(f"SVM Mean Accuracy: {np.mean(svm_scores)}")

    # Cross-Validation for Logistic Regression
    log_reg_scores = cross_val_score(log_reg_model, X, y, cv=5)
    print(f"Logistic Regression Cross-Validation Scores: {log_reg_scores}")
    print(f"Logistic Regression Mean Accuracy: {np.mean(log_reg_scores)}")

    # Cross-Validation for Decision Tree
    decision_tree_scores = cross_val_score(decision_tree_model, X, y, cv=5)
    print(f"Decision Tree Cross-Validation Scores: {decision_tree_scores}")
    print(f"Decision Tree Mean Accuracy: {np.mean(decision_tree_scores)}")

    # Define models to train and evaluate
    models = ['svm', 'logistic_regression', 'decision_tree', 'neural_network']

    for model_type in models:
        print(f"\nTraining and evaluating model: {model_type}")
        accuracy, report, conf_matrix = train_and_evaluate(model_type, X_train, y_train, X_test, y_test)
        print(f"Accuracy for {model_type}: {accuracy}")
        print(f"Classification Report for {model_type}:\n{report}")
        print(f"Confusion Matrix for {model_type}:\n{conf_matrix}")

if __name__ == "__main__":
    main()


