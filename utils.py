"""
Utility Functions
Helper functions for data validation, CSV template generation, etc.
"""

import pandas as pd
import io


def create_template_csv():
    """
    Create a template CSV file for portfolio data.
    
    Returns:
        CSV content as string (for download)
    """
    template_data = {
        'Ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'Shares': [10, 5, 8],
        'Avg_Price': [150.00, 300.00, 2500.00],
        'Currency': ['USD', 'USD', 'USD']
    }
    
    df = pd.DataFrame(template_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


def validate_portfolio_data(df):
    """
    Validate portfolio DataFrame structure and data.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with 'valid' (bool) and 'error' (str) keys
    """
    required_columns = ['Ticker', 'Shares', 'Avg_Price', 'Currency']
    
    # Check if DataFrame is empty
    if df is None or df.empty:
        return {'valid': False, 'error': 'Portfolio data is empty'}
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return {
            'valid': False,
            'error': f'Missing required columns: {", ".join(missing_columns)}'
        }
    
    # Validate data types and values
    for idx, row in df.iterrows():
        # Check Ticker
        if pd.isna(row['Ticker']) or str(row['Ticker']).strip() == '':
            return {'valid': False, 'error': f'Row {idx + 1}: Ticker is required'}
        
        # Check Shares
        try:
            shares = float(row['Shares'])
            if shares <= 0:
                return {'valid': False, 'error': f'Row {idx + 1}: Shares must be greater than 0'}
        except (ValueError, TypeError):
            return {'valid': False, 'error': f'Row {idx + 1}: Shares must be a valid number'}
        
        # Check Avg_Price
        try:
            avg_price = float(row['Avg_Price'])
            if avg_price <= 0:
                return {'valid': False, 'error': f'Row {idx + 1}: Avg_Price must be greater than 0'}
        except (ValueError, TypeError):
            return {'valid': False, 'error': f'Row {idx + 1}: Avg_Price must be a valid number'}
        
        # Check Currency
        valid_currencies = ['SEK', 'USD', 'EUR']
        currency = str(row['Currency']).upper()
        if currency not in valid_currencies:
            return {
                'valid': False,
                'error': f'Row {idx + 1}: Currency must be one of {", ".join(valid_currencies)}'
            }
    
    return {'valid': True, 'error': None}


def format_currency(value, currency='SEK', decimals=2):
    """
    Format a numeric value as currency.
    
    Args:
        value: Numeric value to format
        currency: Currency code
        decimals: Number of decimal places
        
    Returns:
        Formatted string (e.g., "1,234.56 SEK")
    """
    if pd.isna(value):
        return f"N/A {currency}"
    
    return f"{value:,.{decimals}f} {currency}"


def format_percentage(value, decimals=2):
    """
    Format a numeric value as percentage.
    
    Args:
        value: Numeric value to format (e.g., 5.5 for 5.5%)
        decimals: Number of decimal places
        
    Returns:
        Formatted string (e.g., "5.50%")
    """
    if pd.isna(value):
        return "N/A"
    
    return f"{value:.{decimals}f}%"


