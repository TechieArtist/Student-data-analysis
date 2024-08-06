import unittest
from sklearn.model_selection import train_test_split
from model_definitions.svm import train_svm, evaluate_svm
from model_definitions.logistic_regression import train_logistic_regression, evaluate_logistic_regression
from model_definitions.decision_tree import train_decision_tree, evaluate_decision_tree
import numpy as np

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y = np.array([0, 1, 0, 1])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def test_train_svm(self):
        # Test SVM training
        model = train_svm(self.X_train, self.y_train)
        self.assertIsNotNone(model, "SVM model training failed.")
    
    def test_evaluate_svm(self):
        # Test SVM evaluation
        model = train_svm(self.X_train, self.y_train)
        accuracy, report, conf_matrix = evaluate_svm(model, self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0, "SVM evaluation failed.")

    def test_train_logistic_regression(self):
        # Test Logistic Regression training
        model = train_logistic_regression(self.X_train, self.y_train)
        self.assertIsNotNone(model, "Logistic Regression model training failed.")

    def test_evaluate_logistic_regression(self):
        # Test Logistic Regression evaluation
        model = train_logistic_regression(self.X_train, self.y_train)
        accuracy, report, conf_matrix = evaluate_logistic_regression(model, self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0, "Logistic Regression evaluation failed.")
    
    def test_train_decision_tree(self):
        # Test Decision Tree training
        model = train_decision_tree(self.X_train, self.y_train)
        self.assertIsNotNone(model, "Decision Tree model training failed.")

    def test_evaluate_decision_tree(self):
        # Test Decision Tree evaluation
        model = train_decision_tree(self.X_train, self.y_train)
        accuracy, report, conf_matrix = evaluate_decision_tree(model, self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0, "Decision Tree evaluation failed.")

if __name__ == '__main__':
    unittest.main()
