"""
data_preprocessing.py

This module provides functions for data preprocessing and fetching analyst ratings for stocks.
It includes functions to clean historical stock data by filling missing values and to retrieve 
mean analyst ratings for a specified stock ticker.

Functions:
    - clean: Extracts the 'Close' column from the data, resets the index, and fills any missing 
      'Close' values with the previous day's value.
    - get_analyst_ratings: Retrieves the mean target price from analyst ratings for a given stock ticker.
"""

import yfinance as yf
import pandas as pd
# Function to extract 'Date' and 'Close' columns, and fill missing 'Close' values with the previous day's value
def clean(data):

    # Provides overview of columns, data types, and nulls
    print("Overview of dataset:")
    data.info()  

    # Preview the first few rows of the data
    print("\nFirst 5 rows of dataset:")
    print(data.head())  

    # Flatten multi-index columns to make them easier to work with
    data.columns = ['_'.join(col).strip() for col in data.columns.values]

    # Extract 'Close' columns, forward-fill any missing 'Close' values
    close_columns = [col for col in data.columns if 'Close' in col]
    data = data[close_columns]
    data = data.ffill()

    # Reset index to have 'Date' as a column and ensure Date column is in correct format
    data.reset_index(inplace=True)
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Check for and display the sum of missing values
    no_of_missing_values = data.isnull().sum().sum()
    print(f"There are {no_of_missing_values} missing values after forward filling.")

    # Drop any rows with remaining missing values
    data = data.dropna()

    return data


# Function to get mean Analyst Ratings
def get_analyst_ratings(ticker):
    info = yf.Ticker(ticker).info
    return {
        'targetMeanPrice': info.get('targetMeanPrice')
    }