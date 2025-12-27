"""
Backup Data Fetcher Module
Provides alternative data sources as backup for stock prices.
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
import requests


def fetch_price_from_alpha_vantage(ticker, api_key=None):
    """
    Fetch stock price from Alpha Vantage API (backup source).
    Requires free API key from https://www.alphavantage.co/support/#api-key
    
    Args:
        ticker: Stock ticker symbol
        api_key: Alpha Vantage API key (optional, can be set in secrets)
        
    Returns:
        Current price (float) or None if failed
    """
    if not api_key:
        # Try to get from Streamlit secrets
        try:
            api_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", None)
        except:
            pass
    
    if not api_key:
        return None
    
    try:
        # Remove exchange suffix for Alpha Vantage (e.g., .ST)
        clean_ticker = ticker.replace('.ST', '').replace('.OL', '').replace('.CO', '')
        
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': clean_ticker,
            'apikey': api_key
        }
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if 'Global Quote' in data and data['Global Quote']:
            price = float(data['Global Quote'].get('05. price', 0))
            return price if price > 0 else None
        
    except Exception as e:
        print(f"Alpha Vantage error for {ticker}: {str(e)}")
    
    return None


def fetch_price_from_finnhub(ticker, api_key=None):
    """
    Fetch stock price from Finnhub API (backup source).
    Requires free API key from https://finnhub.io/
    
    Args:
        ticker: Stock ticker symbol
        api_key: Finnhub API key (optional, can be set in secrets)
        
    Returns:
        Current price (float) or None if failed
    """
    if not api_key:
        try:
            api_key = st.secrets.get("FINNHUB_API_KEY", None)
        except:
            pass
    
    if not api_key:
        return None
    
    try:
        # Remove exchange suffix for Finnhub
        clean_ticker = ticker.replace('.ST', '').replace('.OL', '').replace('.CO', '')
        
        url = f"https://finnhub.io/api/v1/quote"
        params = {
            'symbol': clean_ticker,
            'token': api_key
        }
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if 'c' in data and data['c'] > 0:  # 'c' is current price
            return float(data['c'])
        
    except Exception as e:
        print(f"Finnhub error for {ticker}: {str(e)}")
    
    return None


def fetch_price_with_fallback(ticker, primary_price=0):
    """
    Try to fetch price from backup sources if primary source failed.
    
    Args:
        ticker: Stock ticker symbol
        primary_price: Price from primary source (yfinance)
        
    Returns:
        Best available price (float)
    """
    # If primary price is valid, use it
    if primary_price and primary_price > 0:
        return primary_price
    
    # Try backup sources
    backup_price = None
    
    # Try Alpha Vantage
    backup_price = fetch_price_from_alpha_vantage(ticker)
    if backup_price and backup_price > 0:
        return backup_price
    
    # Try Finnhub
    backup_price = fetch_price_from_finnhub(ticker)
    if backup_price and backup_price > 0:
        return backup_price
    
    # If all fail, return 0
    return 0


def fetch_price_multiple_methods_yfinance(ticker):
    """
    Try multiple methods from yfinance to get the most reliable price.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Best available price (float)
    """
    from data_fetcher import normalize_ticker
    
    normalized_ticker = normalize_ticker(ticker)
    stock = yf.Ticker(normalized_ticker)
    
    prices = []
    
    # Method 1: info dictionary
    try:
        info = stock.info
        if info:
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            if price and price > 0:
                prices.append(('info', float(price)))
    except:
        pass
    
    # Method 2: fast_info (faster, less data)
    try:
        fast_info = stock.fast_info
        if fast_info:
            price = fast_info.get('lastPrice') or fast_info.get('regularMarketPrice')
            if price and price > 0:
                prices.append(('fast_info', float(price)))
    except:
        pass
    
    # Method 3: history - last minute
    try:
        hist = stock.history(period="1d", interval="1m")
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
            if price > 0:
                prices.append(('history_1m', price))
    except:
        pass
    
    # Method 4: history - last 5 days
    try:
        hist = stock.history(period="5d")
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
            if price > 0:
                prices.append(('history_5d', price))
    except:
        pass
    
    # Return the most recent/reliable price
    if prices:
        # Prefer currentPrice > regularMarketPrice > history
        for method, price in prices:
            if method in ['info', 'fast_info']:
                return price
        # If no info price, return most recent history price
        return prices[-1][1]
    
    return 0

