import unittest
import pandas as pd
import numpy as np
from source.data_preprocessing import load_data, preprocess_data, split_data


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_data = {
            'Age': [20, 25, np.nan, 30, 22],
            'GPA': [3.5, np.nan, 3.0, 3.8, 3.2],
            'Passed': ['Yes', 'No', 'Yes', 'No', 'Yes']
        }
        self.df = pd.DataFrame(self.sample_data)
        self.filepath = 'tests/test_data.csv'
        self.df.to_csv(self.filepath, index=False)
    
    def tearDown(self):
        # Clean up test file
        import os
        os.remove(self.filepath)

    def test_load_data(self):
        # Test loading data from a CSV
        X, y = load_data(self.filepath)
        self.assertEqual(X.shape[0], self.df.shape[0])
        self.assertEqual(X.shape[1], self.df.shape[1] - 1)
        self.assertEqual(len(y), self.df.shape[0])

    def test_handle_missing_values(self):
        # Test if missing values are handled correctly
        processed_df = preprocess_data(self.df.copy())
        self.assertFalse(processed_df['Age'].isnull().any(), "Missing values in 'Age' were not handled.")
        self.assertFalse(processed_df['GPA'].isnull().any(), "Missing values in 'GPA' were not handled.")

    def test_scaling(self):
        # Test if data is scaled correctly
        processed_df = preprocess_data(self.df.copy())
        self.assertTrue((processed_df['Age'] >= 0).all() and (processed_df['Age'] <= 1).all(), "Age not scaled between 0 and 1.")
        self.assertTrue((processed_df['GPA'] >= 0).all() and (processed_df['GPA'] <= 1).all(), "GPA not scaled between 0 and 1.")

    def test_encoding(self):
        # Test if categorical data is encoded correctly
        processed_df = preprocess_data(self.df.copy())
        self.assertTrue(set(processed_df['Passed']) <= {0, 1}, "Categorical values not correctly encoded.")
    
    def test_split_data(self):
        # Test if data is split correctly
        processed_df = preprocess_data(self.df.copy())
        X_train, X_test, y_train, y_test = split_data(processed_df)
        self.assertEqual(len(X_train) + len(X_test), len(processed_df))
        self.assertEqual(len(y_train) + len(y_test), len(processed_df))

if __name__ == '__main__':
    unittest.main()
