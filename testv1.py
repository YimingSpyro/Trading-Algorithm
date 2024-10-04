import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# List of stocks to analyze
tickers = [
    # Technology
    'AAPL', 'META', 'AMZN', 'NVDA', 'MSFT',
    
    # FinTech
    'PYPL', 'AXP', 'MA', 'GPN', 'V',
    
    # Finance
    'GS', 'JPM', 'BLK', 'C', 'BX',
    
    # Consumer
    'KO', 'WMT', 'MCD', 'NKE', 'SBUX'
]

# Function to calculate technical indicators
def calculate_indicators(data):
    # Calculate MACD
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ma = data['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ma
    signal = macd.ewm(span=9, adjust=False).mean()

    # Calculate RSI
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands
    sma = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)

    return macd, signal, rsi, upper_band, lower_band

# Define Buy and Sell signals
def define_signals(data):
    data['Buy_Signal'] = ((data['MACD'] < data['Signal']) & (data['MACD'] < 0) & (data['RSI'] < 30) & (data['Close'] <= data['Lower_Band'])).rolling(window=5).sum() >= 1
    data['Sell_Signal'] = ((data['MACD'] > data['Signal']) & (data['MACD'] > 0) & (data['RSI'] > 70) & (data['Close'] >= data['Upper_Band'])).rolling(window=5).sum() >= 1
    return data

# Function to calculate average sell signals between buy signals
def average_sell_signals(data):
    sell_count = 0
    buy_count = 0
    buy_state = 0
    total_sell_signals_between_buys = 0

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

# Implement trading algorithm
def trading_strategy(data, avg_sell_signals, initial_amount=10000):
    balance = initial_amount
    stock_quantity = 0
    profits = []
    in_trade = False
    buy_price = None
    sell_signal_count = 0

    sell_threshold = avg_sell_signals / 2

    for i in range(len(data)):
        if data['Buy_Signal'].iloc[i] and not in_trade:
            stock_quantity = balance / data['Close'].iloc[i]
            balance = 0
            buy_price = data['Close'].iloc[i]
            in_trade = True
            sell_signal_count = 0 
            
        elif data['Sell_Signal'].iloc[i] and in_trade:
            sell_signal_count += 1
            
            if sell_signal_count >= sell_threshold:
                balance = stock_quantity * data['Close'].iloc[i]
                stock_quantity = 0
                sell_price = data['Close'].iloc[i]
                profit_percent = ((sell_price - buy_price) / buy_price) * 100
                profits.append(profit_percent)
                in_trade = False
                sell_signal_count = 0

    if in_trade:
        balance = stock_quantity * data['Close'].iloc[-1]
    
    final_amount = balance
    return profits, final_amount, in_trade  # Return in_trade status

# Calculate performance metrics
def calculate_performance(profits, initial_amount, final_amount, period_years):
    total_trades = len(profits)
    avg_return_per_trade = sum(profits) / total_trades if total_trades > 0 else 0
    total_return = ((final_amount - initial_amount) / initial_amount) * 100
    avg_annual_return = ((final_amount / initial_amount) ** (1 / period_years) - 1) * 100 if period_years > 0 else 0
    return total_trades, avg_return_per_trade, total_return, avg_annual_return

# Function to fetch analyst ratings
def get_analyst_ratings(ticker):
    info = yf.Ticker(ticker).info
    return {
        'targetMeanPrice': info.get('targetMeanPrice')
    }

# Function to calculate potential upside
def calculate_upside(current_price, target_price):
    if target_price <= 0:
        raise ValueError("Target price must be positive")
    return ((target_price - current_price) / current_price) * 100

# Function to analyze a single stock
def analyze_stock(ticker):
    data = yf.download(ticker, period='10y')

    train_data = data[:-252]  # Use all but the last 252 trading days for training (1 year)
    test_data = data[-252:]   # Reserve the last 252 trading days for testing

    # Apply indicator calculations on training and test data
    train_data['MACD'], train_data['Signal'], train_data['RSI'], train_data['Upper_Band'], train_data['Lower_Band'] = calculate_indicators(train_data)
    test_data['MACD'], test_data['Signal'], test_data['RSI'], test_data['Upper_Band'], test_data['Lower_Band'] = calculate_indicators(test_data)

    # Apply signals to both datasets
    train_data = define_signals(train_data)
    test_data = define_signals(test_data)

    # Calculate average sell signals between buy signals
    avg_sell_signals_train = average_sell_signals(train_data)
    avg_sell_signals_test = average_sell_signals(test_data)

    # Run the strategy on test data
    test_profits, test_final_balance, in_trade_status = trading_strategy(test_data, avg_sell_signals_test)

    # Calculate performance metrics for test data
    test_years = len(test_data) / 252
    test_total_trades, test_avg_return_per_trade, test_total_return, test_avg_annual_return = calculate_performance(
        test_profits, 10000, test_final_balance, test_years)

    # Fetch fundamental data
    info = get_analyst_ratings(ticker)
    target_price = info.get('targetMeanPrice')

    current_price = train_data['Close'].iloc[-1]

    # Calculate potential upside
    if target_price is not None:
        try:
            upside = calculate_upside(current_price, target_price)
        except ValueError:
            upside = None
    else:
        upside = None

    return {
        'Ticker': ticker,
        'Current Price': round(current_price, 2),
        'Target Price': round(target_price, 2) if target_price is not None else None,
        'Potential Upside (%)': round(upside, 2) if upside is not None else None,
        'Trades Completed': test_total_trades,
        'Average Return per Trade (%)': round(test_avg_return_per_trade, 2),
        'Total Return (%)': round(test_total_return, 2),
        'In Trade': in_trade_status  # Status of trade
    }

# Analyze all stocks
results = []
for ticker in tickers:
    result = analyze_stock(ticker)
    results.append(result)

# Create DataFrame
df = pd.DataFrame(results)

# Split into two tables: Stock Ranking and Trade Bot Performance
stock_ranking_df = df[['Ticker', 'Current Price', 'Target Price', 'Potential Upside (%)']].sort_values(by='Potential Upside (%)', ascending=False)

# Add a Rank column
stock_ranking_df['Rank'] = range(1, len(stock_ranking_df) + 1)

# Reorder the original DataFrame based on the ranking order
df = df.set_index('Ticker')
ranked_tickers = stock_ranking_df['Ticker'].values
trade_bot_performance_df = df.loc[ranked_tickers, ['Trades Completed', 'Average Return per Trade (%)', 'Total Return (%)', 'In Trade']]

# Streamlit App
st.title("Stock Analysis and Trading Bot")

# Display Stock Ranking Table
st.subheader("Stock Ranking")
st.write("*Current Price refers to the last closing price of training dataset.")
st.write("*Target Price refers to the mean analyst price target.")
st.dataframe(stock_ranking_df[['Rank', 'Ticker', 'Current Price', 'Target Price', 'Potential Upside (%)']])

# Add Status column
trade_bot_performance_df['Status'] = trade_bot_performance_df['In Trade'].apply(lambda x: 'In Trade' if x else 'Not In Trade')

# Rearrange columns: Shift status column to the right of Trades Completed
trade_bot_performance_df = trade_bot_performance_df[['Trades Completed', 'Average Return per Trade (%)', 'Total Return (%)', 'Status']]

# Display Trade Bot Performance Table
st.subheader("Trade Bot Performance")
st.write("*Trading performance based on 1 year of test data.")
st.dataframe(trade_bot_performance_df)
