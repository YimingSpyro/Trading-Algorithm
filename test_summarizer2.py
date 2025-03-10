import streamlit as st
import yfinance as yf
from datetime import datetime

# Streamlit app title and description
st.title("Quarterly Financial Statements Viewer")
st.write("Enter a stock ticker to view the latest quarterly financial statements.")

# Input for stock ticker
ticker_input = st.text_input("Stock Ticker", value="AAPL", max_chars=10)

def summarize_income_statement(income_statement):
    """Summarize key figures from the income statement."""
    try:
        revenue = income_statement.loc['Total Revenue'][0]
        gross_profit = income_statement.loc['Gross Profit'][0]
        operating_income = income_statement.loc['Operating Income'][0]
        net_income = income_statement.loc['Net Income'][0]
        eps = income_statement.loc['Earnings Per Share'][0]
        
        st.write(f"**Revenue**: {revenue:,.2f}")
        st.write(f"**Gross Profit**: {gross_profit:,.2f}")
        st.write(f"**Operating Income**: {operating_income:,.2f}")
        st.write(f"**Net Income**: {net_income:,.2f}")
        st.write(f"**Earnings Per Share (EPS)**: {eps:,.2f}")
        
        # Net profit margin
        net_margin = (net_income / revenue) * 100 if revenue != 0 else 0
        st.write(f"**Net Profit Margin**: {net_margin:.2f}%")
        
    except Exception as e:
        st.write("Error summarizing income statement:", e)

def summarize_balance_sheet(balance_sheet):
    """Summarize key figures from the balance sheet."""
    try:
        total_assets = balance_sheet.loc['Total Assets'][0]
        total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'][0]
        total_equity = balance_sheet.loc['Total Stockholder Equity'][0]
        
        st.write(f"**Total Assets**: {total_assets:,.2f}")
        st.write(f"**Total Liabilities**: {total_liabilities:,.2f}")
        st.write(f"**Total Equity**: {total_equity:,.2f}")
        
        # Debt-to-equity ratio
        debt_to_equity = total_liabilities / total_equity if total_equity != 0 else 0
        st.write(f"**Debt-to-Equity Ratio**: {debt_to_equity:.2f}")
        
    except Exception as e:
        st.write("Error summarizing balance sheet:", e)

def summarize_cashflow(cashflow):
    """Summarize key figures from the cash flow statement."""
    try:
        operating_cashflow = cashflow.loc['Total Cash From Operating Activities'][0]
        investing_cashflow = cashflow.loc['Total Cash From Investing Activities'][0]
        financing_cashflow = cashflow.loc['Total Cash From Financing Activities'][0]
        
        st.write(f"**Operating Cash Flow**: {operating_cashflow:,.2f}")
        st.write(f"**Investing Cash Flow**: {investing_cashflow:,.2f}")
        st.write(f"**Financing Cash Flow**: {financing_cashflow:,.2f}")
        
        # Free cash flow (simplified)
        free_cash_flow = operating_cashflow + investing_cashflow
        st.write(f"**Free Cash Flow**: {free_cash_flow:,.2f}")
        
    except Exception as e:
        st.write("Error summarizing cash flow statement:", e)

if ticker_input:
    # Retrieve stock data using yfinance
    stock = yf.Ticker(ticker_input)
    
    # Get the last report date
    last_report_date = stock.quarterly_financials.columns[0]
    formatted_date = datetime.strptime(str(last_report_date).split()[0], '%Y-%m-%d').strftime('%B %Y')
    
    # Displaying which quarterly report is shown
    st.header(f"Latest Quarterly Report: {formatted_date}")

    st.write("""This report provides a breakdown of the financial performance and position for the selected company during the latest reported quarter. The statements are divided into three sections: Income Statement, Balance Sheet, and Cash Flow Statement.""")

    # Income Statement
    st.subheader("Income Statement")
    st.write("""The income statement shows the company's revenues, expenses, and net income for the period. It reflects the company's profitability by detailing its revenue from sales, cost of goods sold, operating expenses, and earnings per share (EPS). Positive net income indicates a profit, while negative net income indicates a loss.""")
    try:
        income_statement = stock.quarterly_financials
        st.write(income_statement)
        summarize_income_statement(income_statement)
    except Exception as e:
        st.write("Income statement data is unavailable.")

    # Balance Sheet
    st.subheader("Balance Sheet")
    st.write("""The balance sheet provides a snapshot of the company’s assets, liabilities, and equity at the end of the quarter. Assets represent what the company owns, liabilities represent what it owes, and equity represents shareholder ownership. This statement helps assess the company's financial stability and liquidity.""")
    try:
        balance_sheet = stock.quarterly_balance_sheet
        st.write(balance_sheet)
        summarize_balance_sheet(balance_sheet)
    except Exception as e:
        st.write("Balance sheet data is unavailable.")

    # Cash Flow Statement
    st.subheader("Cash Flow Statement")
    st.write("""The cash flow statement shows the movement of cash in and out of the company over the quarter. It is divided into operating, investing, and financing activities. Positive cash flow from operations indicates that the company generates enough cash to maintain and grow its operations.""")
    try:
        cashflow = stock.quarterly_cashflow
        st.write(cashflow)
        summarize_cashflow(cashflow)
    except Exception as e:
        st.write("Cash flow statement data is unavailable.")
