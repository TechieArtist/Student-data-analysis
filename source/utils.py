import numpy as np
import matplotlib.pyplot as plt
import os

def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_data(file_path):
    """
    Loads data from a CSV file.
    
    Args:
    - file_path (str): Path to the CSV file.
    
    Returns:
    - X (numpy array): Features.
    - y (numpy array): Labels.
    """
    try:
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        X = data[:, :-1]  # All rows, all but last column
        y = data[:, -1]   # All rows, last column
        return X, y
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None, None

def plot_data(X, y, pos_label="Positive", neg_label="Negative", pos_marker='+', neg_marker='o', pos_color='blue', neg_color='red'):
    """
    Plots data points X and y into a new figure.
    Plots the positive examples with + and the negative examples with o.
    
    Args:
    - X (numpy array): Data features (m x 2 matrix).
    - y (numpy array): Data labels (m x 1 vector).
    - pos_label (str): Label for positive examples.
    - neg_label (str): Label for negative examples.
    - pos_marker (str): Marker style for positive examples.
    - neg_marker (str): Marker style for negative examples.
    - pos_color (str): Color for positive examples.
    - neg_color (str): Color for negative examples.
    """
    pos = y == 1
    neg = y == 0

    plt.scatter(X[pos, 0], X[pos, 1], marker=pos_marker, c=pos_color, label=pos_label)
    plt.scatter(X[neg, 0], X[neg, 1], marker=neg_marker, c=neg_color, label=neg_label)

def plot_decision_boundary(w, b, X, y):
    """
    Plots the decision boundary learned by the logistic regression model.
    
    Args:
    - w (numpy array): Weights of the logistic regression model.
    - b (float): Bias of the logistic regression model.
    - X (numpy array): Data features (m x 2 matrix).
    - y (numpy array): Data labels (m x 1 vector).
    """
    plot_data(X, y)

    # Compute the decision boundary line
    if w[1] != 0:
        x_values = [np.min(X[:, 0]), np.max(X[:, 0])]
        y_values = -(w[0] * np.array(x_values) + b) / w[1]
        plt.plot(x_values, y_values, label='Decision Boundary')
    else:
        plt.axvline(x=-b/w[0], color='k', linestyle='--', label='Decision Boundary')
    
    plt.legend()
    plt.show()
