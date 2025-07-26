import time
import pandas as pd
import numpy as np
import requests
import re
import os
import io
import base64

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.black_litterman import BlackLittermanModel

import matplotlib.pyplot as plt
import seaborn as sns

# confg
plt.style.use('dark_background')


# Load Data

def parse_tickers_from_text(text_input):

    if not text_input:
        return []
    tickers = [ticker.strip().upper() for ticker in text_input.split(',')]
    return list(filter(None, tickers))  # Remove any empty strings


def parse_uploaded_csv(file_stream):

    #parsing and userinputs

    df = pd.read_csv(file_stream)

    ticker_col = next((col for col in df.columns if re.search(r'ticker|symbol|asset', col, re.I)), None)
    if not ticker_col:
        raise ValueError("CSV must contain a 'Ticker', 'Symbol', or 'Asset' column.")
    df.rename(columns={ticker_col: 'Ticker'}, inplace=True)
    df['Ticker'] = df['Ticker'].str.strip().str.upper()

    esg_cols = {
        'E': next((col for col in df.columns if re.search(r'e_score|environment', col, re.I)), None),
        'S': next((col for col in df.columns if re.search(r's_score|social', col, re.I)), None),
        'G': next((col for col in df.columns if re.search(r'g_score|governance', col, re.I)), None),
        'Total': next((col for col in df.columns if re.search(r'esg_score|total|esg risk', col, re.I)), None)
    }

    esg_data = df[['Ticker']].copy()
    has_esg = False
    for key, col_name in esg_cols.items():
        if col_name:
            esg_data[key] = df[col_name]
            has_esg = True

    if not has_esg:
        return df['Ticker'].tolist(), None

    return df['Ticker'].tolist(), esg_data


def get_financial_data(tickers, api_key):
    #Fetches historical adjusted close prices from Alpha Vantage
    print(f" Starting financial data fetch for {len(tickers)} tickers. ")
    all_prices = pd.DataFrame()
    fetched_tickers = []

    for ticker in tickers:
        print(f"\nProcessing ticker: {ticker}")
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "compact",
            "apikey": api_key
        }
        try:
            print(f"Making API call for {ticker}...")
            response = requests.get("https://www.alphavantage.co/query", params=params, timeout=15)
            data = response.json()
            print(f"Server response for {ticker}: {data}")  # error detection

            if "Time Series (Daily)" in data:
                print(f"Successfully received data for {ticker}.")
                prices = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                prices = prices['5. adjusted close'].astype(float)
                all_prices[ticker] = prices
                fetched_tickers.append(ticker)
            else:
                #ALpha vintage status
                print(f"Warning: Could not find 'Time Series (Daily)' in response for {ticker}.")

            # delay for the 5 calls/minute API
            print("Pausing for 12 seconds ")
            time.sleep(12)

        except Exception as e:
            print(f"An exception occurred while fetching data for {ticker}: {e}")
            continue

    if all_prices.empty:
        # successful completion but no result
        raise ValueError("Could not fetch financial data for any of the provided tickers after trying all of them.")

    all_prices.index = pd.to_datetime(all_prices.index)
    all_prices.sort_index(inplace=True)
    excluded_stocks = list(set(tickers) - set(fetched_tickers))
    return all_prices.dropna(), excluded_stocks

#2. Clustering Logic
def perform_clustering(data_for_clustering, method):
    if data_for_clustering.shape[0] < 3:
        data_for_clustering['Cluster'] = 0
        return data_for_clustering

    if method == 'Group by Similarity': #K-Means
        model = KMeans(n_clusters=min(3, data_for_clustering.shape[0]), random_state=42, n_init=10)
    elif method == 'Flexible Grouping ':
        model = GaussianMixture(n_components=min(3, data_for_clustering.shape[0]), random_state=42)
    elif method == 'DBSCAN':
        model = DBSCAN(eps=0.5, min_samples=2)
    else:
        data_for_clustering['Cluster'] = 'N/A'
        return data_for_clustering

    labels = model.fit_predict(data_for_clustering)
    data_for_clustering['Cluster'] = labels
    return data_for_clustering


#  Optimization
def run_optimization(prices, strategy):
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    if strategy == "Mean-Variance": #MPT
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
    elif strategy == "HRP":
        hrp = HRPOpt(prices.pct_change().dropna())
        weights = hrp.optimize()
    elif strategy == "Black-Litterman":
        # Simplified example, may fail if specified stocks aren't in portfolio
        try:
            tickers = list(mu.index)
            views = {}
            if "GOOGL" in tickers and "MSFT" in tickers:
                views = pd.Series({"GOOGL": 0.05, "MSFT": -0.05})  # View: GOOGL will outperform MSFT

            bl = BlackLittermanModel(S, pi=mu, absolute_views=views, omega='idzorek')
            mu_bl = bl.bl_returns()
            S_bl = bl.bl_cov()
            ef_bl = EfficientFrontier(mu_bl, S_bl)
            weights = ef_bl.max_sharpe()
        except Exception as e:
            print(f"Black-Litterman failed: {e}. Falling back to Mean Variance.")
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe()
    elif strategy == "EqualWeighting":
        num_assets = len(prices.columns)
        weights = {ticker: 1 / num_assets for ticker in prices.columns}
    else:
        raise ValueError(f"Unknown optimization strategy: {strategy}")


    if isinstance(weights, dict):
        return weights
    ef = EfficientFrontier(mu, S)
    ef.set_weights(weights)
    return dict(ef.clean_weights())   # clean weights = rounds and removes tiny weights


# Plotting and Output Generation
def generate_plot(plot_function, **kwargs):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_function(ax=ax, **kwargs)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def plot_pie_chart(ax, weights):
    filtered_weights = {k: v for k, v in weights.items() if v > 0.001}
    labels = filtered_weights.keys()
    sizes = filtered_weights.values()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title('Optimized Portfolio Allocation', color='white')               #no weight stocks excluded for cleaner chart view


def plot_esg_distribution(ax, esg_data):
    if 'Total' in esg_data.columns:
        esg_scores = esg_data.set_index('Ticker')['Total'].rename('ESG Score')
    elif 'E_Score' in esg_data.columns:
        esg_scores = esg_data.set_index('Ticker')[['E_Score', 'S_Score', 'G_Score']].mean(axis=1).rename('ESG Score')
    else:
        return
    sns.barplot(x=esg_scores.index, y=esg_scores.values, ax=ax, palette='viridis')
    ax.set_title('Overall ESG Score Distribution', color='white')
    ax.set_ylabel('ESG Score')
    ax.tick_params(axis='x', rotation=45)


def format_df_to_html(df):
    return df.to_html(classes=['table', 'table-striped', 'table-dark', 'table-hover'], border=0)
