"""
Data Fetcher Module
Handles fetching stock data and currency exchange rates from yfinance.
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta


def normalize_ticker(ticker):
    """
    Normalize ticker symbol for yfinance.
    Handles different exchange suffixes like .ST (Stockholm), .OL (Oslo), etc.
    
    Args:
        ticker: Ticker symbol (e.g., 'investor-b.st', 'AAPL')
        
    Returns:
        Normalized ticker for yfinance
    """
    ticker = ticker.upper().strip()
    
    # Handle Stockholm exchange (.ST)
    if ticker.endswith('.ST'):
        # yfinance uses .ST suffix as-is
        return ticker
    elif ticker.endswith('.STO'):
        return ticker.replace('.STO', '.ST')
    
    # Handle other common exchanges
    # Oslo (.OL)
    if ticker.endswith('.OL'):
        return ticker
    
    # Copenhagen (.CO)
    if ticker.endswith('.CO'):
        return ticker
    
    # For US stocks, ensure no suffix issues
    if '.' in ticker and not any(ticker.endswith(suffix) for suffix in ['.ST', '.OL', '.CO', '.DE', '.L']):
        # Might be a ticker with dot, try as-is first
        pass
    
    return ticker


@st.cache_data(ttl=60)  # Cache for 1 minute (shorter to allow manual updates)
def fetch_stock_data(portfolio_df, use_manual_prices=True):
    """
    Fetch current stock data for all tickers in the portfolio.
    Uses manually entered prices if available, otherwise fetches from yfinance.
    
    Args:
        portfolio_df: DataFrame with 'Ticker' column
        use_manual_prices: If True, use manually entered prices from database
        
    Returns:
        DataFrame with stock data including current price, dividend info, etc.
    """
    if portfolio_df is None or portfolio_df.empty:
        return None
    
    # Get manually entered prices if available
    manual_prices = {}
    if use_manual_prices:
        try:
            from database import get_manual_price_data, calculate_annual_dividend_from_dividends
            manual_df = get_manual_price_data()
            for _, row in manual_df.iterrows():
                ticker = str(row['ticker']).upper()  # Normalize to uppercase
                current_price = row.get('current_price', 0)
                
                # Calculate annual dividend from dividend records first
                annual_from_dividends = calculate_annual_dividend_from_dividends(ticker)
                
                # Use dividend records if available, otherwise use manual annual_dividend
                annual_dividend = annual_from_dividends if annual_from_dividends > 0 else row.get('annual_dividend', 0)
                
                if pd.notna(current_price) and float(current_price) > 0:
                    manual_prices[ticker] = {
                        'current_price': float(current_price),
                        'annual_dividend': float(annual_dividend) if pd.notna(annual_dividend) and float(annual_dividend) > 0 else None,
                        'source': 'manual'
                    }
        except Exception as e:
            print(f"Error loading manual prices: {str(e)}")
    
    # Get user's currency for each ticker from portfolio_df
    user_currency_map = {}
    if 'Currency' in portfolio_df.columns:
        for _, row in portfolio_df.iterrows():
            user_currency_map[str(row['Ticker']).upper()] = row['Currency']
    
    tickers = portfolio_df['Ticker'].unique().tolist()
    stock_data = []
    
    for ticker in tickers:
        # Check if manual price exists (normalize ticker for comparison)
        ticker_upper = str(ticker).upper()
        if ticker_upper in manual_prices:
            manual_data = manual_prices[ticker_upper]
            # Get user's currency for this ticker
            user_currency = user_currency_map.get(ticker_upper, 'USD')
            
            try:
                # Still get other info from yfinance (company name, etc.)
                normalized_ticker = normalize_ticker(ticker)
                stock = yf.Ticker(normalized_ticker)
                info = stock.info
                
                stock_data.append({
                    'Ticker': ticker,
                    'Company_Name': info.get('longName') or info.get('shortName') or ticker,
                    'Current_Price': manual_data['current_price'],
                    'Price_Source': 'manual',
                    'Annual_Dividend': manual_data['annual_dividend'] or 0,
                    'Dividend_Yield_Info': 0,
                    'Week_52_High': info.get('fiftyTwoWeekHigh', 0) or 0,
                    'Week_52_Low': info.get('fiftyTwoWeekLow', 0) or 0,
                    'Market_Cap': info.get('marketCap', 0) or 0,
                    'Payout_Ratio': info.get('payoutRatio', 0) or 0,
                    'Currency': user_currency  # Use user's currency, not yfinance currency
                })
                continue  # Skip yfinance price fetching
            except Exception as e:
                # If yfinance fails, use manual data only
                stock_data.append({
                    'Ticker': ticker,
                    'Company_Name': ticker,
                    'Current_Price': manual_data['current_price'],
                    'Price_Source': 'manual',
                    'Annual_Dividend': manual_data['annual_dividend'] or 0,
                    'Dividend_Yield_Info': 0,
                    'Week_52_High': 0,
                    'Week_52_Low': 0,
                    'Market_Cap': 0,
                    'Payout_Ratio': 0,
                    'Currency': user_currency  # Use user's currency
                })
                continue
        
        # If no manual price, fetch from yfinance
        try:
            # Normalize ticker for yfinance (handle .ST, .OL, etc.)
            normalized_ticker = normalize_ticker(ticker)
            stock = yf.Ticker(normalized_ticker)
            info = stock.info
            
            # Get current price - try multiple methods for better reliability
            current_price = 0
            price_source = "none"
            
            # Method 1: Try info dictionary (most reliable)
            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose') or 0
            if current_price > 0:
                price_source = "yfinance_info"
            
            # Method 2: Try fast_info (faster, less data but sometimes more reliable)
            if current_price == 0:
                try:
                    fast_info = stock.fast_info
                    if fast_info:
                        fast_price = fast_info.get('lastPrice') or fast_info.get('regularMarketPrice')
                        if fast_price and fast_price > 0:
                            current_price = float(fast_price)
                            price_source = "yfinance_fast_info"
                except:
                    pass
            
            # Method 3: If info doesn't have price, try getting from history (1 minute)
            if current_price == 0:
                try:
                    hist = stock.history(period="1d", interval="1m")
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                        price_source = "yfinance_history_1m"
                except:
                    pass
            
            # Method 4: Try last close price from daily history (5 days)
            if current_price == 0:
                try:
                    hist = stock.history(period="5d")
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                        price_source = "yfinance_history_5d"
                except:
                    pass
            
            # Method 5: Try backup sources if yfinance failed
            if current_price == 0:
                try:
                    from data_fetcher_backup import fetch_price_with_fallback
                    backup_price = fetch_price_with_fallback(ticker, 0)
                    if backup_price and backup_price > 0:
                        current_price = backup_price
                        price_source = "backup_source"
                except ImportError:
                    # Backup module not available, skip
                    pass
                except Exception as e:
                    print(f"Backup source error for {ticker}: {str(e)}")
            
            # Get dividend information
            dividend_rate = info.get('dividendRate', 0) or 0
            trailing_annual_dividend_rate = info.get('trailingAnnualDividendRate', 0) or 0
            annual_dividend = dividend_rate if dividend_rate > 0 else trailing_annual_dividend_rate
            
            # Get dividend yield
            dividend_yield = info.get('dividendYield', 0) or 0
            
            # Get company name
            company_name = info.get('longName') or info.get('shortName') or ticker
            
            # Get 52-week high/low
            week_52_high = info.get('fiftyTwoWeekHigh', 0) or 0
            week_52_low = info.get('fiftyTwoWeekLow', 0) or 0
            
            # Get market cap
            market_cap = info.get('marketCap', 0) or 0
            
            # Get payout ratio
            payout_ratio = info.get('payoutRatio', 0) or 0
            
            # Get currency from yfinance, but we'll override with user's currency later
            yf_currency = info.get('currency', 'USD')
            
            stock_data.append({
                'Ticker': ticker,
                'Company_Name': company_name,
                'Current_Price': current_price,
                'Price_Source': price_source,  # Track which source provided the price
                'Annual_Dividend': annual_dividend,
                'Dividend_Yield_Info': dividend_yield * 100 if dividend_yield else 0,
                'Week_52_High': week_52_high,
                'Week_52_Low': week_52_low,
                'Market_Cap': market_cap,
                'Payout_Ratio': payout_ratio,
                'Currency': yf_currency  # This will be overridden by user's currency from database
            })
            
        except Exception as e:
            # If fetching fails, add placeholder data
            stock_data.append({
                'Ticker': ticker,
                'Company_Name': ticker,
                'Current_Price': 0,
                'Price_Source': 'failed',
                'Annual_Dividend': 0,
                'Dividend_Yield_Info': 0,
                'Week_52_High': 0,
                'Week_52_Low': 0,
                'Market_Cap': 0,
                'Payout_Ratio': 0,
                'Currency': 'USD'
            })
            print(f"Error fetching data for {ticker}: {str(e)}")
    
    return pd.DataFrame(stock_data)


@st.cache_data(ttl=3600)  # Cache for 1 hour (currency rates change less frequently)
def fetch_currency_rates():
    """
    Fetch current currency exchange rates for SEK/USD and SEK/EUR.
    Also calculates inverse rates and cross rates.
    
    Returns:
        Dictionary with currency pair rates (e.g., {'SEK/USD': 0.095, 'USD/SEK': 10.5, ...})
    """
    rates = {}
    
    try:
        # Fetch SEK/USD rate
        sek_usd = yf.Ticker("SEKUSD=X")
        sek_usd_data = sek_usd.history(period="1d")
        if not sek_usd_data.empty:
            sek_usd_rate = float(sek_usd_data['Close'].iloc[-1])
            rates['SEK/USD'] = sek_usd_rate
            rates['USD/SEK'] = 1 / sek_usd_rate if sek_usd_rate > 0 else 1
    except Exception as e:
        print(f"Error fetching SEK/USD: {str(e)}")
        # Fallback rates (approximate)
        rates['SEK/USD'] = 0.095
        rates['USD/SEK'] = 10.5
    
    try:
        # Fetch SEK/EUR rate
        sek_eur = yf.Ticker("SEKEUR=X")
        sek_eur_data = sek_eur.history(period="1d")
        if not sek_eur_data.empty:
            sek_eur_rate = float(sek_eur_data['Close'].iloc[-1])
            rates['SEK/EUR'] = sek_eur_rate
            rates['EUR/SEK'] = 1 / sek_eur_rate if sek_eur_rate > 0 else 1
    except Exception as e:
        print(f"Error fetching SEK/EUR: {str(e)}")
        # Fallback rates (approximate)
        rates['SEK/EUR'] = 0.090
        rates['EUR/SEK'] = 11.1
    
    try:
        # Fetch USD/EUR rate (for cross-rate calculations)
        usd_eur = yf.Ticker("USDEUR=X")
        usd_eur_data = usd_eur.history(period="1d")
        if not usd_eur_data.empty:
            usd_eur_rate = float(usd_eur_data['Close'].iloc[-1])
            rates['USD/EUR'] = usd_eur_rate
            rates['EUR/USD'] = 1 / usd_eur_rate if usd_eur_rate > 0 else 1
    except Exception as e:
        print(f"Error fetching USD/EUR: {str(e)}")
        # Calculate from SEK rates if available
        if 'SEK/USD' in rates and 'SEK/EUR' in rates:
            rates['USD/EUR'] = rates['SEK/EUR'] / rates['SEK/USD']
            rates['EUR/USD'] = 1 / rates['USD/EUR']
        else:
            rates['USD/EUR'] = 0.95
            rates['EUR/USD'] = 1.05
    
    # Add same-currency rates (always 1.0)
    rates['SEK/SEK'] = 1.0
    rates['USD/USD'] = 1.0
    rates['EUR/EUR'] = 1.0
    
    return rates


def get_currency_rate(from_currency, to_currency, rates_dict):
    """
    Get exchange rate between two currencies.
    
    Args:
        from_currency: Source currency code
        to_currency: Target currency code
        rates_dict: Dictionary of currency rates
        
    Returns:
        Exchange rate (float)
    """
    if from_currency == to_currency:
        return 1.0
    
    # Try direct rate
    rate_key = f"{from_currency}/{to_currency}"
    if rate_key in rates_dict:
        return rates_dict[rate_key]
    
    # Try inverse rate
    inverse_key = f"{to_currency}/{from_currency}"
    if inverse_key in rates_dict:
        return 1 / rates_dict[inverse_key]
    
    # Try cross-rate via USD
    if from_currency != 'USD' and to_currency != 'USD':
        usd_key1 = f"{from_currency}/USD"
        usd_key2 = f"USD/{to_currency}"
        if usd_key1 in rates_dict and usd_key2 in rates_dict:
            return rates_dict[usd_key1] * rates_dict[usd_key2]
    
    # Default fallback
    return 1.0

