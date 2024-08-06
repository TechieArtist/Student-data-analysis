import pandas as pd

def get_summary_statistics(df):
    return df.describe()

def calculate_correlations(df):
    return df.corr()
