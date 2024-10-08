import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

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
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ma = data['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ma
    signal = macd.ewm(span=9, adjust=False).mean()

    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    sma = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)

    return macd, signal, rsi, upper_band, lower_band

# Function to identify Buy and Sell Signals
def define_signals(data):
    data.loc[:, 'Buy_Signal'] = ((data['MACD'] < data['Signal']) & (data['MACD'] < 0) & (data['RSI'] < 30) & (data['Close'] <= data['Lower_Band'])).rolling(window=5).sum() >= 1
    data.loc[:, 'Sell_Signal'] = ((data['MACD'] > data['Signal']) & (data['MACD'] > 0) & (data['RSI'] > 70) & (data['Close'] >= data['Upper_Band'])).rolling(window=5).sum() >= 1
    return data

# Function to calculate average sell signals
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

# Function for Trading algorithm
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

# Function to calculate/record trading strategy performance
def calculate_performance(profits, initial_amount, final_amount, period_years):
    total_trades = len(profits)
    avg_return_per_trade = sum(profits) / total_trades if total_trades > 0 else 0
    total_return = ((final_amount - initial_amount) / initial_amount) * 100
    avg_annual_return = ((final_amount / initial_amount) ** (1 / period_years) - 1) * 100 if period_years > 0 else 0
    return total_trades, avg_return_per_trade, total_return, avg_annual_return

# Function to get mean Analyst Ratings
def get_analyst_ratings(ticker):
    info = yf.Ticker(ticker).info
    return {
        'targetMeanPrice': info.get('targetMeanPrice')
    }

# Function to calculate potential upside based on analyst target price/current price
def calculate_upside(current_price, target_price):
    if target_price <= 0:
        raise ValueError("Target price must be positive")
    return ((target_price - current_price) / current_price) * 100

# Display trading algorithm performance
def analyze_stock(ticker):
    data = yf.download(ticker, period='10y')

    train_data = data[:-252]
    test_data = data[-252:]

    train_data['MACD'], train_data['Signal'], train_data['RSI'], train_data['Upper_Band'], train_data['Lower_Band'] = calculate_indicators(train_data)
    test_data['MACD'], test_data['Signal'], test_data['RSI'], test_data['Upper_Band'], test_data['Lower_Band'] = calculate_indicators(test_data)

    train_data = define_signals(train_data)
    test_data = define_signals(test_data)

    avg_sell_signals_train = average_sell_signals(train_data)
    avg_sell_signals_test = average_sell_signals(test_data)

    test_profits, test_final_balance, in_trade = trading_strategy(test_data, avg_sell_signals_test)

    test_years = len(test_data) / 252
    test_total_trades, test_avg_return_per_trade, test_total_return, test_avg_annual_return = calculate_performance(
        test_profits, 10000, test_final_balance, test_years)

    info = get_analyst_ratings(ticker)
    target_price = info.get('targetMeanPrice')
    current_price = train_data['Close'].iloc[-1]
        
    if target_price is not None:
        try:
            upside = calculate_upside(current_price, target_price)
        except ValueError:
            upside = None
    else:
        upside = None

    return {
        'Ticker': ticker,
        'Current Price ($)': round(current_price, 2),
        'Target Price ($)': round(target_price, 2) if target_price is not None else None,
        'Potential Upside (%)': round(upside, 2) if upside is not None else None,
        'Trades Closed': test_total_trades,
        'Average Return per Trade (%)': round(test_avg_return_per_trade, 2),
        'Total Return (%)': round(test_total_return, 2),
        'In Trade': in_trade,  # Indicate if still in trade
        'Train Data': train_data,
        'Test Data': test_data
    }

# Function to save results to a PDF
def save_to_pdf(results):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)

    # Create PDF content
    text = c.beginText(30, height - 40)
    text.setFont("Helvetica", 12)
    text.setLeading(15)  # Set line spacing

    # Add each stock's analysis results to the PDF
    for result in results:
        text.textLine(f"Ticker: {result['Ticker']}")
        text.textLine(f"Current Price: ${result['Current Price ($)']}")
        text.textLine(f"Target Price: ${result['Target Price ($)']}")
        text.textLine(f"Potential Upside: {result['Potential Upside (%)']}%")
        text.textLine(f"Trades Closed: {result['Trades Closed']}")
        text.textLine(f"Average Return per Trade: {result['Average Return per Trade (%)']}%")
        text.textLine(f"Total Return: {result['Total Return (%)']}%")
        text.textLine(f"In Trade: {'Yes' if result['In Trade'] else 'No'}")
        text.textLine("")  # Add a blank line for spacing

    c.drawText(text)
    c.showPage()
    c.save()

    # Save the PDF
    buffer.seek(0)
    return buffer

# Streamlit app
st.title("Stock Trading Strategy Analyzer")
st.write("Analyze stocks and visualize trading strategies.")

selected_stocks = st.multiselect("Select stocks to analyze:", tickers)

results = []
if st.button("Analyze"):
    for stock in selected_stocks:
        result = analyze_stock(stock)
        results.append(result)

    if results:
        st.write("Analysis Results:")
        for result in results:
            st.write(result)

        # Create PDF button
        if st.button("Save Results to PDF"):
            pdf_buffer = save_to_pdf(results)
            st.download_button("Download PDF", pdf_buffer, "stock_analysis_results.pdf", "application/pdf")
