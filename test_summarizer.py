import yfinance as yf
from transformers import pipeline
import streamlit as st

# Initialize summarizer pipeline from Hugging Face (choose an appropriate model for summaries)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Streamlit app layout
st.title("Quarterly Financial Statement Summarizer")
st.write("Enter a stock ticker to view and summarize quarterly financial statements.")

# Input for stock ticker
ticker_input = st.text_input("Stock Ticker", value="AAPL", max_chars=10)

def get_financial_summary(data, section_name):
    # Convert the data to text for summarization
    data_str = data.to_string()
    st.subheader(f"{section_name} - Raw Data")
    st.write(data)

    # Summarize using Hugging Face pipeline
    st.subheader(f"{section_name} - Summary")
    summary = summarizer(data_str, max_length=300, min_length=0, do_sample=False)[0]["summary_text"]
    st.write(summary)

if ticker_input:
    stock = yf.Ticker(ticker_input)
    st.header(f"Latest Quarterly Report Summary for {ticker_input}")

    # Get and summarize income statement
    income_statement = stock.quarterly_financials
    if not income_statement.empty:
        get_financial_summary(income_statement, "Income Statement")

    # Get and summarize balance sheet
    balance_sheet = stock.quarterly_balance_sheet
    if not balance_sheet.empty:
        get_financial_summary(balance_sheet, "Balance Sheet")

    # Get and summarize cash flow statement
    cashflow = stock.quarterly_cashflow
    if not cashflow.empty:
        get_financial_summary(cashflow, "Cash Flow Statement")
