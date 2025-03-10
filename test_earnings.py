import streamlit as st
import yfinance as yf
from datetime import datetime

# Streamlit app title and description
st.title("Quarterly Financial Statements Viewer")
st.write("Enter a stock ticker to view the latest quarterly financial statements.")

# Input for stock ticker
ticker_input = st.text_input("Stock Ticker", value="AAPL", max_chars=10)

if ticker_input:
    # Retrieve stock data using yfinance
    stock = yf.Ticker(ticker_input)
    
    # Get the last report date
    last_report_date = stock.quarterly_financials.columns[0]
    formatted_date = datetime.strptime(str(last_report_date).split()[0], '%Y-%m-%d').strftime('%B %Y')
    
    # Displaying which quarterly report is shown
    st.header(f"Latest Quarterly Report: {formatted_date}")

    st.write("""
    This report provides a breakdown of the financial performance and position for the selected company during the latest reported quarter. 
    The statements are divided into three sections: Income Statement, Balance Sheet, and Cash Flow Statement.
    """)

    # Income Statement
    st.subheader("Income Statement")
    st.write("""
    The income statement shows the company's revenues, expenses, and net income for the period. 
    It reflects the company's profitability by detailing its revenue from sales, cost of goods sold, operating expenses, 
    and earnings per share (EPS). Positive net income indicates a profit, while negative net income indicates a loss.
    """)
    try:
        income_statement = stock.quarterly_financials
        st.write(income_statement)
    except Exception as e:
        st.write("Income statement data is unavailable.")

    # Balance Sheet
    st.subheader("Balance Sheet")
    st.write("""
    The balance sheet provides a snapshot of the company’s assets, liabilities, and equity at the end of the quarter. 
    Assets represent what the company owns, liabilities represent what it owes, and equity represents shareholder ownership. 
    This statement helps assess the company's financial stability and liquidity.
    """)
    try:
        balance_sheet = stock.quarterly_balance_sheet
        st.write(balance_sheet)
    except Exception as e:
        st.write("Balance sheet data is unavailable.")

    # Cash Flow Statement
    st.subheader("Cash Flow Statement")
    st.write("""
    The cash flow statement shows the movement of cash in and out of the company over the quarter. 
    It is divided into operating, investing, and financing activities. Positive cash flow from operations 
    indicates that the company generates enough cash to maintain and grow its operations.
    """)
    try:
        cashflow = stock.quarterly_cashflow
        st.write(cashflow)
    except Exception as e:
        st.write("Cash flow statement data is unavailable.")
