import unittest
import pandas as pd
from source.data_preprocessing import load_data, preprocess_data, split_data
import numpy as np




class TestDataCleaning(unittest.TestCase):
    
    def setUp(self):
        # This function will run before each test case
        self.sample_data = {
            'Age': [20, 25, np.nan, 30, 22],
            'GPA': [3.5, np.nan, 3.0, 3.8, 3.2],
            'Passed': ['Yes', 'No', 'Yes', 'No', 'Yes']
        }
        self.df = pd.DataFrame(self.sample_data)
        self.filepath = 'tests/test_data.csv'
        self.df.to_csv(self.filepath, index=False)
    
    def tearDown(self):
        # This function will run after each test case
        import os
        os.remove(self.filepath)

    def test_load_data(self):
        # Test if data is loaded correctly
        df = load_data(self.filepath)
        self.assertEqual(df.shape[0], self.df.shape[0])
        self.assertEqual(df.shape[1], self.df.shape[1])

    def test_preprocess_data(self):
        # Test if data is preprocessed correctly
        processed_df = preprocess_data(self.df.copy())
        self.assertFalse(processed_df['Age'].isnull().any(), "Age column still has missing values.")
        self.assertFalse(processed_df['GPA'].isnull().any(), "GPA column still has missing values.")
        self.assertTrue((processed_df['GPA'] <= 1).all() and (processed_df['GPA'] >= 0).all(), "GPA column not scaled properly.")
        self.assertTrue(processed_df['Passed'].dtype == int, "Passed column not encoded properly.")
    
    def test_split_data(self):
        # Test if data is split correctly
        processed_df = preprocess_data(self.df.copy())
        X_train, X_test, y_train, y_test = split_data(processed_df)
        self.assertEqual(len(X_train) + len(X_test), len(processed_df))
        self.assertEqual(len(y_train) + len(y_test), len(processed_df))

if __name__ == '__main__':
    unittest.main()
