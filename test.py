import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF
import os

# List of stocks to analyze
tickers = [
    'AAPL', 'META', 'AMZN', 'NVDA', 'MSFT', 'TSLA', 'GOOG', 'GOOGL',
    'PYPL', 'AXP', 'MA', 'GPN', 'V', 'GS', 'JPM', 'BLK', 'C', 'BX',
    'KO', 'WMT', 'MCD', 'NKE', 'SBUX'
]

# Calculate technical indicators
def calculate_indicators(data):
    data['ShortEMA'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['LongEMA'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['ShortEMA'] - data['LongEMA']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    sma = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = sma + (std_dev * 2)
    data['Lower_Band'] = sma - (std_dev * 2)
    return data

# Define buy and sell signals
def define_signals(data):
    data['Buy_Signal'] = ((data['MACD'] < data['Signal']) & (data['MACD'] < 0) & (data['RSI'] < 30) & (data['Close'] <= data['Lower_Band'])).rolling(window=5).sum() >= 1
    data['Sell_Signal'] = ((data['MACD'] > data['Signal']) & (data['MACD'] > 0) & (data['RSI'] > 70) & (data['Close'] >= data['Upper_Band'])).rolling(window=5).sum() >= 1
    return data

# Calculate average sell signals between buys
def average_sell_signals(data):
    sell_count, buy_count, buy_state, total_sell_signals_between_buys = 0, 0, 0, 0

    for i in range(len(data)):
        if data['Buy_Signal'].iloc[i]:
            buy_state += 1
            if buy_state >= 1:
                total_sell_signals_between_buys += sell_count
                sell_count = 0  
        elif data['Sell_Signal'].iloc[i]:
            if buy_state >= 1:
                buy_state = 0
                buy_count += 1
            sell_count += 1

    average = total_sell_signals_between_buys / buy_count if buy_count > 0 else 0
    return average

# Trading strategy based on signals
def trading_strategy(data, avg_sell_signals, initial_amount=10000):
    balance, stock_quantity, profits = initial_amount, 0, []
    in_trade, buy_price, sell_signal_count = False, None, 0
    sell_threshold = avg_sell_signals / 2

    for i in range(len(data)):
        if data['Buy_Signal'].iloc[i] and not in_trade:
            stock_quantity = balance / data['Close'].iloc[i]
            balance, in_trade, sell_signal_count = 0, True, 0
            buy_price = data['Close'].iloc[i]
        elif data['Sell_Signal'].iloc[i] and in_trade:
            sell_signal_count += 1
            if sell_signal_count >= sell_threshold:
                balance = stock_quantity * data['Close'].iloc[i]
                profit_percent = ((balance / (stock_quantity * buy_price)) - 1) * 100
                profits.append(profit_percent)
                stock_quantity, in_trade, sell_signal_count = 0, False, 0

    if in_trade:
        balance = stock_quantity * data['Close'].iloc[-1]
    
    return profits, balance, in_trade

# Calculate trading performance metrics
def calculate_performance(profits, initial_amount, final_amount, period_years):
    total_trades = len(profits)
    avg_return_per_trade = sum(profits) / total_trades if total_trades > 0 else 0
    total_return = ((final_amount - initial_amount) / initial_amount) * 100
    avg_annual_return = ((final_amount / initial_amount) ** (1 / period_years) - 1) * 100 if period_years > 0 else 0
    return total_trades, avg_return_per_trade, total_return, avg_annual_return

# Get analyst target price for potential upside calculation
def get_analyst_ratings(ticker):
    info = yf.Ticker(ticker).info
    return info.get('targetMeanPrice')

# Calculate upside potential
def calculate_upside(current_price, target_price):
    if target_price <= 0:
        raise ValueError("Target price must be positive")
    return ((target_price - current_price) / current_price) * 100

# Data cleaning
def clean(data):
    data = data[['Close']]
    data.reset_index(inplace=True)
    data['Close'].fillna(method='ffill', inplace=True)
    return data

# Analyze stock and calculate metrics
def analyze_stock(ticker):
    data = clean(yf.download(ticker, period='10y'))
    data = calculate_indicators(data)
    data = define_signals(data)
    avg_sell_signals = average_sell_signals(data)

    test_profits, test_final_balance, in_trade = trading_strategy(data, avg_sell_signals)

    years = len(data) / 252
    trades, avg_return_per_trade, total_return, avg_annual_return = calculate_performance(test_profits, 10000, test_final_balance, years)

    target_price = get_analyst_ratings(ticker)
    current_price = data['Close'].iloc[-1]
    upside = calculate_upside(current_price, target_price) if target_price else None

    return {
        'Ticker': ticker,
        'Current Price ($)': round(current_price, 2),
        'Target Price ($)': round(target_price, 2) if target_price else None,
        'Potential Upside (%)': round(upside, 2) if upside else None,
        'Total Trades': trades,
        'Avg Return per Trade (%)': round(avg_return_per_trade, 2),
        'Total Return (%)': round(total_return, 2),
        'In Trade': in_trade
    }

# Plot stock data and save/display
def plot_or_save_stock_data(ticker, data, action='show'):
    data = data.copy()
    data['Date'] = pd.to_datetime(data.index)
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(14, 10), sharex=True)

    # Plot Bollinger Bands and signals
    ax1.plot(data['Close'], color='black', label='Close Price')
    ax1.plot(data['Upper_Band'], color='red', label='Upper Bollinger Band')
    ax1.plot(data['Lower_Band'], color='blue', label='Lower Bollinger Band')
    ax1.scatter(data.index[data['Buy_Signal']], data['Close'][data['Buy_Signal']], marker='^', color='green', label='Buy Signal')
    ax1.scatter(data.index[data['Sell_Signal']], data['Close'][data['Sell_Signal']], marker='v', color='red', label='Sell Signal')
    ax1.set_title(f'{ticker} Stock Analysis')
    ax1.legend()

    ax2.plot(data['MACD'], label='MACD', color='green')
    ax2.plot(data['Signal'], label='Signal Line', color='red')
    ax2.axhline(0, color='black', linestyle='--')
    ax2.legend()

    ax3.plot(data['RSI'], label='RSI', color='purple')
    ax3.axhline(30, linestyle='--', color='red')
    ax3.axhline(70, linestyle='--', color='red')
    ax3.legend()

    if action == 'save':
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{ticker}_analysis.png')
        plt.close()
    else:
        st.pyplot(fig)
        plt.close()

# Generate PDF report
def download_results_as_pdf(stock_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Stock Analysis Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)

    for key, value in stock_data.items():
        pdf.cell(0, 10, f'{key}: {value}', ln=True)
    
    os.makedirs("reports", exist_ok=True)
    pdf.output("reports/Stock_Analysis_Report.pdf")

# Streamlit interface
st.title("Automated Stock Analysis")

# User selection of stock
stock = st.selectbox("Choose a Stock Ticker", tickers)
if st.button("Analyze Stock"):
    stock_data = analyze_stock(stock)
    plot_or_save_stock_data(stock, yf.download(stock, period='10y'), action='show')
    st.write(stock_data)
    
    if st.button("Download Report"):
        download_results_as_pdf(stock_data)
        st.success("Report downloaded successfully.")
