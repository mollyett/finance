"""
Portfolio Calculator Module
Handles portfolio calculations including gains, losses, and dividend metrics.
"""

import pandas as pd
from datetime import datetime
from data_fetcher import get_currency_rate


def calculate_portfolio_metrics(portfolio_data, user_data, base_currency, currency_rates):
    """
    Calculate overall portfolio metrics.
    
    Args:
        portfolio_data: DataFrame with stock data from yfinance
        user_data: DataFrame with user's portfolio (Ticker, Shares, Avg_Price, Currency)
        base_currency: Base currency for calculations
        currency_rates: Dictionary of currency exchange rates
        
    Returns:
        Dictionary with portfolio metrics
    """
    metrics = {
        'total_value': 0,
        'total_cost': 0,
        'total_gain_loss': 0,
        'gain_loss_percent': 0,
        'annual_dividend_income': 0,
        'total_value_change': 0
    }
    
    for idx, user_row in user_data.iterrows():
        ticker = user_row['Ticker']
        shares = user_row['Shares']
        avg_price = user_row['Avg_Price']
        stock_currency = user_row['Currency']
        
        # Find matching stock data
        stock_data = portfolio_data[portfolio_data['Ticker'] == ticker]
        if stock_data.empty:
            continue
        
        stock_row = stock_data.iloc[0]
        current_price = stock_row.get('Current_Price', 0)
        
        # Convert prices to base currency
        price_rate = get_currency_rate(stock_currency, base_currency, currency_rates)
        current_price_base = current_price * price_rate
        avg_price_base = avg_price * price_rate
        
        # Calculate values
        market_value = shares * current_price_base
        cost_basis = shares * avg_price_base
        gain_loss = market_value - cost_basis
        
        metrics['total_value'] += market_value
        metrics['total_cost'] += cost_basis
        metrics['total_gain_loss'] += gain_loss
        
        # Calculate dividend income
        annual_dividend = stock_row.get('Annual_Dividend', 0)
        if pd.notna(annual_dividend) and annual_dividend > 0:
            dividend_rate = get_currency_rate(stock_currency, base_currency, currency_rates)
            annual_dividend_base = annual_dividend * dividend_rate
            metrics['annual_dividend_income'] += shares * annual_dividend_base
    
    # Calculate percentage gain/loss
    if metrics['total_cost'] > 0:
        metrics['gain_loss_percent'] = (metrics['total_gain_loss'] / metrics['total_cost']) * 100
    
    return metrics


def calculate_dividend_yield(annual_dividend, current_price, stock_currency, base_currency, currency_rates):
    """
    Calculate dividend yield in base currency.
    
    Args:
        annual_dividend: Annual dividend per share
        current_price: Current stock price
        stock_currency: Currency of the stock
        base_currency: Base currency for calculation
        currency_rates: Dictionary of currency exchange rates
        
    Returns:
        Dividend yield as percentage
    """
    if current_price <= 0 or annual_dividend <= 0:
        return 0.0
    
    # Convert to base currency
    rate = get_currency_rate(stock_currency, base_currency, currency_rates)
    dividend_base = annual_dividend * rate
    price_base = current_price * rate
    
    if price_base > 0:
        yield_pct = (dividend_base / price_base) * 100
        return round(yield_pct, 2)
    
    return 0.0


def calculate_stock_metrics(ticker, shares, avg_price, stock_currency, portfolio_data, base_currency, currency_rates):
    """
    Calculate metrics for a single stock.
    
    Args:
        ticker: Stock ticker symbol
        shares: Number of shares owned
        avg_price: Average purchase price
        stock_currency: Currency of the stock
        portfolio_data: DataFrame with stock data
        base_currency: Base currency for calculations
        currency_rates: Dictionary of currency exchange rates
        
    Returns:
        Dictionary with stock-specific metrics
    """
    stock_data = portfolio_data[portfolio_data['Ticker'] == ticker]
    if stock_data.empty:
        return None
    
    stock_row = stock_data.iloc[0]
    current_price = stock_row.get('Current_Price', 0)
    annual_dividend = stock_row.get('Annual_Dividend', 0)
    
    # Convert to base currency
    rate = get_currency_rate(stock_currency, base_currency, currency_rates)
    current_price_base = current_price * rate
    avg_price_base = avg_price * rate
    
    # Calculate metrics
    market_value = shares * current_price_base
    cost_basis = shares * avg_price_base
    gain_loss = market_value - cost_basis
    gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
    
    dividend_yield = calculate_dividend_yield(
        annual_dividend if pd.notna(annual_dividend) else 0,
        current_price,
        stock_currency,
        base_currency,
        currency_rates
    )
    
    annual_dividend_income = 0
    if pd.notna(annual_dividend) and annual_dividend > 0:
        annual_dividend_base = annual_dividend * rate
        annual_dividend_income = shares * annual_dividend_base
    
    return {
        'ticker': ticker,
        'market_value': market_value,
        'cost_basis': cost_basis,
        'gain_loss': gain_loss,
        'gain_loss_pct': gain_loss_pct,
        'dividend_yield': dividend_yield,
        'annual_dividend_income': annual_dividend_income,
        'current_price_base': current_price_base,
        'avg_price_base': avg_price_base
    }


def calculate_snowball_effect(portfolio_data, user_data, base_currency, currency_rates, years=10, reinvest_dividends=True, annual_growth_rate=0.05):
    """
    Calculate the snowball effect of dividend reinvestment over time.
    
    Args:
        portfolio_data: DataFrame with stock data from yfinance
        user_data: DataFrame with user's portfolio
        base_currency: Base currency for calculations
        currency_rates: Dictionary of currency exchange rates
        years: Number of years to project
        reinvest_dividends: Whether to reinvest dividends
        annual_growth_rate: Expected annual stock price growth rate (default 5%)
        
    Returns:
        DataFrame with year-by-year projections
    """
    projections = []
    
    # Calculate initial values
    initial_value = 0
    initial_dividend_income = 0
    
    for idx, user_row in user_data.iterrows():
        ticker = user_row['Ticker']
        shares = user_row['Shares']
        stock_currency = user_row['Currency']
        
        stock_data = portfolio_data[portfolio_data['Ticker'] == ticker]
        if stock_data.empty:
            continue
        
        stock_row = stock_data.iloc[0]
        current_price = stock_row.get('Current_Price', 0)
        annual_dividend = stock_row.get('Annual_Dividend', 0) or 0
        
        rate = get_currency_rate(stock_currency, base_currency, currency_rates)
        current_price_base = current_price * rate
        annual_dividend_base = annual_dividend * rate if pd.notna(annual_dividend) else 0
        
        initial_value += shares * current_price_base
        initial_dividend_income += shares * annual_dividend_base
    
    # Project year by year
    current_portfolio_value = initial_value
    total_dividends_received = 0
    total_shares = {row['Ticker']: row['Shares'] for _, row in user_data.iterrows()}
    
    for year in range(years + 1):
        year_dividend_income = 0
        new_shares_from_dividends = {}
        
        # Calculate dividend income for this year
        for idx, user_row in user_data.iterrows():
            ticker = user_row['Ticker']
            stock_currency = user_row['Currency']
            
            stock_data = portfolio_data[portfolio_data['Ticker'] == ticker]
            if stock_data.empty:
                continue
            
            stock_row = stock_data.iloc[0]
            annual_dividend = stock_row.get('Annual_Dividend', 0) or 0
            
            if pd.notna(annual_dividend) and annual_dividend > 0:
                rate = get_currency_rate(stock_currency, base_currency, currency_rates)
                annual_dividend_base = annual_dividend * rate
                
                # Use current shares (including reinvested)
                current_shares = total_shares.get(ticker, user_row['Shares'])
                dividend_income = current_shares * annual_dividend_base
                year_dividend_income += dividend_income
                
                # Reinvest dividends if enabled
                if reinvest_dividends:
                    current_price = stock_row.get('Current_Price', 0)
                    current_price_base = current_price * rate
                    if current_price_base > 0:
                        new_shares = dividend_income / current_price_base
                        new_shares_from_dividends[ticker] = new_shares
                        total_shares[ticker] = total_shares.get(ticker, user_row['Shares']) + new_shares
        
        # Apply growth to portfolio value
        if year > 0:
            current_portfolio_value *= (1 + annual_growth_rate)
        
        # Add reinvested dividends to portfolio value
        if reinvest_dividends:
            for ticker, new_shares in new_shares_from_dividends.items():
                stock_data = portfolio_data[portfolio_data['Ticker'] == ticker]
                if not stock_data.empty:
                    stock_row = stock_data.iloc[0]
                    user_row = user_data[user_data['Ticker'] == ticker].iloc[0]
                    rate = get_currency_rate(user_row['Currency'], base_currency, currency_rates)
                    current_price_base = stock_row.get('Current_Price', 0) * rate
                    current_portfolio_value += new_shares * current_price_base
        
        total_dividends_received += year_dividend_income
        
        projections.append({
            'Year': year,
            'Portfolio_Value': current_portfolio_value,
            'Annual_Dividend_Income': year_dividend_income,
            'Total_Dividends_Received': total_dividends_received,
            'Cumulative_Return': ((current_portfolio_value - initial_value) / initial_value * 100) if initial_value > 0 else 0
        })
    
    return pd.DataFrame(projections)


def calculate_advanced_statistics(portfolio_data, user_data, base_currency, currency_rates):
    """
    Calculate advanced portfolio statistics.
    
    Args:
        portfolio_data: DataFrame with stock data
        user_data: DataFrame with user's portfolio
        base_currency: Base currency
        currency_rates: Currency exchange rates
        
    Returns:
        Dictionary with advanced statistics
    """
    stats = {
        'num_positions': len(user_data),
        'total_shares': 0,
        'weighted_avg_dividend_yield': 0,
        'largest_position': None,
        'smallest_position': None,
        'top_dividend_payers': [],
        'portfolio_concentration': {}
    }
    
    position_values = []
    dividend_data = []
    total_value = 0
    
    for idx, user_row in user_data.iterrows():
        ticker = user_row['Ticker']
        shares = user_row['Shares']
        stock_currency = user_row['Currency']
        
        stock_data = portfolio_data[portfolio_data['Ticker'] == ticker]
        if stock_data.empty:
            continue
        
        stock_row = stock_data.iloc[0]
        current_price = stock_row.get('Current_Price', 0)
        annual_dividend = stock_row.get('Annual_Dividend', 0) or 0
        
        rate = get_currency_rate(stock_currency, base_currency, currency_rates)
        current_price_base = current_price * rate
        market_value = shares * current_price_base
        
        stats['total_shares'] += shares
        total_value += market_value
        
        position_values.append({
            'ticker': ticker,
            'value': market_value,
            'shares': shares
        })
        
        if pd.notna(annual_dividend) and annual_dividend > 0:
            annual_dividend_base = annual_dividend * rate
            dividend_yield = calculate_dividend_yield(
                annual_dividend,
                current_price,
                stock_currency,
                base_currency,
                currency_rates
            )
            dividend_income = shares * annual_dividend_base
            
            dividend_data.append({
                'ticker': ticker,
                'dividend_yield': dividend_yield,
                'dividend_income': dividend_income,
                'weight': market_value
            })
    
    # Find largest and smallest positions
    if position_values:
        largest = max(position_values, key=lambda x: x['value'])
        smallest = min(position_values, key=lambda x: x['value'])
        stats['largest_position'] = {
            'ticker': largest['ticker'],
            'value': largest['value'],
            'percentage': (largest['value'] / total_value * 100) if total_value > 0 else 0
        }
        stats['smallest_position'] = {
            'ticker': smallest['ticker'],
            'value': smallest['value'],
            'percentage': (smallest['value'] / total_value * 100) if total_value > 0 else 0
        }
    
    # Calculate weighted average dividend yield
    if dividend_data and total_value > 0:
        weighted_yield = sum(
            (d['dividend_income'] / total_value * 100) for d in dividend_data
        )
        stats['weighted_avg_dividend_yield'] = weighted_yield
    
    # Top dividend payers
    if dividend_data:
        top_payers = sorted(dividend_data, key=lambda x: x['dividend_income'], reverse=True)[:5]
        stats['top_dividend_payers'] = [
            {
                'ticker': d['ticker'],
                'dividend_income': d['dividend_income'],
                'dividend_yield': d['dividend_yield']
            }
            for d in top_payers
        ]
    
    # Portfolio concentration (top 5 holdings)
    if position_values:
        sorted_positions = sorted(position_values, key=lambda x: x['value'], reverse=True)
        top_5_value = sum(p['value'] for p in sorted_positions[:5])
        stats['portfolio_concentration'] = {
            'top_5_percentage': (top_5_value / total_value * 100) if total_value > 0 else 0,
            'top_5_tickers': [p['ticker'] for p in sorted_positions[:5]]
        }
    
    return stats


def calculate_yield_on_cost(portfolio_data, user_data, base_currency, currency_rates):
    """
    Calculate Yield on Cost (YoY) for each holding.
    YoY = Annual Dividend per Share / Average Purchase Price
    
    Args:
        portfolio_data: DataFrame with stock data
        user_data: DataFrame with user's portfolio
        base_currency: Base currency
        currency_rates: Currency exchange rates
        
    Returns:
        Dictionary with YoY for each ticker
    """
    yoc_data = {}
    
    for idx, user_row in user_data.iterrows():
        ticker = user_row['Ticker']
        avg_price = user_row['Avg_Price']
        stock_currency = user_row['Currency']
        
        stock_data = portfolio_data[portfolio_data['Ticker'] == ticker]
        if stock_data.empty:
            continue
        
        stock_row = stock_data.iloc[0]
        annual_dividend = stock_row.get('Annual_Dividend', 0) or 0
        
        if pd.notna(annual_dividend) and annual_dividend > 0 and avg_price > 0:
            rate = get_currency_rate(stock_currency, base_currency, currency_rates)
            dividend_base = annual_dividend * rate
            price_base = avg_price * rate
            
            if price_base > 0:
                yoc = (dividend_base / price_base) * 100
                yoc_data[ticker] = yoc
    
    return yoc_data


def calculate_dividend_cagr(transactions_df, portfolio_data, base_currency, currency_rates):
    """
    Calculate Compound Annual Growth Rate (CAGR) for dividends.
    Requires transaction history with dates.
    
    Args:
        transactions_df: DataFrame with transaction history
        portfolio_data: DataFrame with current stock data
        base_currency: Base currency
        currency_rates: Currency exchange rates
        
    Returns:
        Dictionary with CAGR for each ticker (if enough data available)
    """
    cagr_data = {}
    
    # Group transactions by ticker
    for ticker in transactions_df['ticker'].unique():
        ticker_transactions = transactions_df[transactions_df['ticker'] == ticker].sort_values('purchase_date')
        
        if len(ticker_transactions) < 2:
            continue  # Need at least 2 transactions to calculate growth
        
        # Get current dividend
        stock_data = portfolio_data[portfolio_data['Ticker'] == ticker]
        if stock_data.empty:
            continue
        
        current_dividend = stock_data.iloc[0].get('Annual_Dividend', 0) or 0
        if not pd.notna(current_dividend) or current_dividend <= 0:
            continue
        
        # Get first transaction date and estimate initial dividend (simplified)
        first_date = pd.to_datetime(ticker_transactions.iloc[0]['purchase_date'])
        years = (datetime.now() - first_date).days / 365.25
        
        if years < 1:
            continue  # Need at least 1 year of history
        
        # Simplified CAGR calculation (assuming dividend growth)
        # In reality, you'd need historical dividend data
        # This is a placeholder that would need historical data
        cagr_data[ticker] = {
            'cagr': None,  # Would need historical dividend data
            'years': years,
            'current_dividend': current_dividend
        }
    
    return cagr_data


def calculate_rebalance_suggestions(portfolio_data, user_data, base_currency, currency_rates):
    """
    Calculate rebalancing suggestions based on target allocations.
    
    Args:
        portfolio_data: DataFrame with stock data
        user_data: DataFrame with user's portfolio (must have Target_Allocation column)
        base_currency: Base currency
        currency_rates: Currency exchange rates
        
    Returns:
        DataFrame with rebalancing suggestions
    """
    suggestions = []
    total_portfolio_value = 0
    
    # Calculate current values and total portfolio value
    holding_values = {}
    for idx, user_row in user_data.iterrows():
        ticker = user_row['Ticker']
        shares = user_row['Shares']
        stock_currency = user_row['Currency']
        target_alloc = user_row.get('Target_Allocation', None)
        
        stock_data = portfolio_data[portfolio_data['Ticker'] == ticker]
        if stock_data.empty:
            continue
        
        stock_row = stock_data.iloc[0]
        current_price = stock_row.get('Current_Price', 0)
        rate = get_currency_rate(stock_currency, base_currency, currency_rates)
        current_price_base = current_price * rate
        market_value = shares * current_price_base
        
        total_portfolio_value += market_value
        holding_values[ticker] = {
            'current_value': market_value,
            'shares': shares,
            'current_price': current_price_base,
            'target_alloc': target_alloc
        }
    
    # Calculate suggestions
    for ticker, data in holding_values.items():
        current_value = data['current_value']
        current_weight = (current_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
        target_weight = data['target_alloc'] if pd.notna(data['target_alloc']) else None
        current_price = data['current_price']
        
        if target_weight is not None and current_price > 0:
            target_value = total_portfolio_value * (target_weight / 100)
            difference = target_value - current_value
            shares_to_buy = difference / current_price if difference > 0 else 0
            shares_to_sell = abs(difference) / current_price if difference < 0 else 0
            
            suggestions.append({
                'Ticker': ticker,
                'Current_Weight_%': round(current_weight, 2),
                'Target_Weight_%': round(target_weight, 2),
                'Difference_%': round(target_weight - current_weight, 2),
                'Current_Value': current_value,
                'Target_Value': target_value,
                'Value_Difference': difference,
                'Shares_to_Buy': round(shares_to_buy, 2) if shares_to_buy > 0 else 0,
                'Shares_to_Sell': round(shares_to_sell, 2) if shares_to_sell > 0 else 0
            })
        else:
            suggestions.append({
                'Ticker': ticker,
                'Current_Weight_%': round(current_weight, 2),
                'Target_Weight_%': target_weight if target_weight else 'N/A',
                'Difference_%': 'N/A',
                'Current_Value': current_value,
                'Target_Value': 'N/A',
                'Value_Difference': 'N/A',
                'Shares_to_Buy': 'N/A',
                'Shares_to_Sell': 'N/A'
            })
    
    return pd.DataFrame(suggestions)

