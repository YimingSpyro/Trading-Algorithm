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
    print("First 5 rows of dataset:")
    data.head()  

    # Extract 'Close' columns, and fill missing 'Close' values with the previous day's value
    data = data[['Close']]
    data['Close'].fillna(method='ffill', inplace=True)

    # Extract 'Date' columns, reset index to have 'Date' as a column and ensure that the Date column is in the correct format
    data.reset_index(inplace=True) 
    data['Date'] = pd.to_datetime(data['Date'])

    # Check for the sum of missing values
    no_of_missing_values=data.isnull().sum()
    print('There is',no_of_missing_values,"missing values")

    # Drops all rows with any missing values
    data = data.dropna()
    return data

# Function to get mean Analyst Ratings
def get_analyst_ratings(ticker):
    info = yf.Ticker(ticker).info
    return {
        'targetMeanPrice': info.get('targetMeanPrice')
    }