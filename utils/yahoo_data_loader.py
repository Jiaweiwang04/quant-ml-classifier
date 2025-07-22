import yfinance as yf
import pandas as pd
import os
import time

tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ', 'PG', 'UNH', 'MA', 'HD', 'BAC', 'XOM', 'PFE', 'DIS', 'KO', 'PEP', 'AVGO', 'ADBE', 'MRK', 'T', 'INTC', 'CSCO', 'CMCSA', 'WMT', 'ABBV', 'NFLX', 'QCOM', 'NKE', 'TMO', 'LLY', 'ACN', 'AMD', 'COST', 'DHR', 'MDT', 'ABT', 'TXN', 'HON', 'CRM', 'ORCL', 'CVX', 'UPS', 'PM', 'UNP', 'MS', 'LIN', 'GS', 'IBM', 'RTX', 'BA', 'BLK', 'CAT', 'GE', 'AMAT', 'INTU', 'LOW', 'ISRG', 'ZTS', 'MDLZ', 'NOW', 'DE', 'SPGI', 'PLD', 'SYK', 'CB', 'GILD', 'ADI', 'AXP', 'CI', 'MO', 'VRTX', 'MMC', 'ADP', 'CL', 'TGT', 'USB', 'BDX', 'LRCX', 'MU', 'SCHW', 'SO', 'FIS', 'PNC', 'TJX', 'APD', 'GM', 'EL', 'C', 'DUK', 'FDX', 'BKNG', 'AON', 'EW', 'CSX', 'MCD', 'ETN']

def download_and_save_per_stock(start="2024-01-01", end="2024-06-01", folder="data/"):
    os.makedirs(folder, exist_ok=True)

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, interval='1d', progress=False, threads=False)
            time.sleep(1)
            if not df.empty:
                df.to_csv(f"{folder}{ticker}.csv")
                print(f"Saved {ticker} data to {folder}{ticker}.csv")
            else:
                print(f"No data for {ticker}")
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")

if __name__ == "__main__":
    download_and_save_per_stock()