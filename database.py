"""
Database Module
Handles SQLite database operations for portfolio transactions and holdings.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
import streamlit as st


def get_db_path():
    """Get the path to the SQLite database file"""
    db_dir = Path(".data")
    db_dir.mkdir(exist_ok=True)
    return db_dir / "portfolio.db"


def init_database():
    """Initialize the database with required tables"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Transactions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            purchase_date DATE NOT NULL,
            purchase_price REAL NOT NULL,
            quantity REAL NOT NULL,
            currency TEXT NOT NULL,
            sector TEXT,
            country TEXT,
            target_allocation REAL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Holdings summary table (for quick access)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            ticker TEXT PRIMARY KEY,
            total_shares REAL NOT NULL,
            avg_price REAL NOT NULL,
            currency TEXT NOT NULL,
            sector TEXT,
            country TEXT,
            target_allocation REAL,
            current_price REAL,
            annual_dividend REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Add new columns if they don't exist (for existing databases)
    try:
        cursor.execute("ALTER TABLE holdings ADD COLUMN current_price REAL")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute("ALTER TABLE holdings ADD COLUMN annual_dividend REAL")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Dividends table for multiple dividends per year with dates
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dividends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            dividend_date DATE NOT NULL,
            dividend_amount REAL NOT NULL,
            currency TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_dividends_ticker_date 
        ON dividends(ticker, dividend_date)
    """)
    
    conn.commit()
    conn.close()


def add_transaction(ticker, purchase_date, purchase_price, quantity, currency, 
                   sector=None, country=None, target_allocation=None, notes=None):
    """Add a new transaction to the database"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO transactions 
        (ticker, purchase_date, purchase_price, quantity, currency, sector, country, target_allocation, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ticker.upper(), purchase_date, purchase_price, quantity, currency, 
          sector, country, target_allocation, notes))
    
    # Update holdings summary
    update_holdings_summary(ticker, purchase_price, quantity, currency, sector, country, target_allocation, conn)
    
    conn.commit()
    conn.close()
    return cursor.lastrowid


def update_holdings_summary(ticker, purchase_price, quantity, currency, sector, country, target_allocation, conn):
    """Update or insert holdings summary"""
    cursor = conn.cursor()
    
    # Check if ticker exists
    cursor.execute("SELECT total_shares, avg_price FROM holdings WHERE ticker = ?", (ticker.upper(),))
    existing = cursor.fetchone()
    
    if existing:
        # Update existing holding (weighted average price)
        old_shares, old_avg_price = existing
        new_total_shares = old_shares + quantity
        new_avg_price = ((old_shares * old_avg_price) + (quantity * purchase_price)) / new_total_shares
        
        cursor.execute("""
            UPDATE holdings 
            SET total_shares = ?, avg_price = ?, currency = ?, 
                sector = COALESCE(?, sector), country = COALESCE(?, country),
                target_allocation = COALESCE(?, target_allocation),
                last_updated = CURRENT_TIMESTAMP
            WHERE ticker = ?
        """, (new_total_shares, new_avg_price, currency, sector, country, target_allocation, ticker.upper()))
    else:
        # Insert new holding
        cursor.execute("""
            INSERT INTO holdings (ticker, total_shares, avg_price, currency, sector, country, target_allocation)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ticker.upper(), quantity, purchase_price, currency, sector, country, target_allocation))


def get_all_transactions():
    """Get all transactions from the database"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    
    df = pd.read_sql_query("""
        SELECT * FROM transactions 
        ORDER BY purchase_date DESC, created_at DESC
    """, conn)
    
    conn.close()
    return df


def get_all_holdings():
    """Get all holdings summary from the database"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    
    df = pd.read_sql_query("""
        SELECT * FROM holdings 
        ORDER BY ticker
    """, conn)
    
    conn.close()
    return df


def get_holdings_as_portfolio_df():
    """Get holdings in the format expected by portfolio calculator"""
    holdings_df = get_all_holdings()
    if holdings_df.empty:
        return pd.DataFrame(columns=['Ticker', 'Shares', 'Avg_Price', 'Currency', 'Sector', 'Country', 'Target_Allocation'])
    
    # Rename columns to match expected format (handle both lowercase and mixed case)
    column_mapping = {}
    if 'ticker' in holdings_df.columns:
        column_mapping['ticker'] = 'Ticker'
    if 'total_shares' in holdings_df.columns:
        column_mapping['total_shares'] = 'Shares'
    if 'avg_price' in holdings_df.columns:
        column_mapping['avg_price'] = 'Avg_Price'
    if 'currency' in holdings_df.columns:
        column_mapping['currency'] = 'Currency'
    if 'sector' in holdings_df.columns:
        column_mapping['sector'] = 'Sector'
    if 'country' in holdings_df.columns:
        column_mapping['country'] = 'Country'
    if 'target_allocation' in holdings_df.columns:
        column_mapping['target_allocation'] = 'Target_Allocation'
    
    holdings_df = holdings_df.rename(columns=column_mapping)
    
    # Ensure all required columns exist, fill missing ones with None
    required_columns = ['Ticker', 'Shares', 'Avg_Price', 'Currency', 'Sector', 'Country', 'Target_Allocation']
    for col in required_columns:
        if col not in holdings_df.columns:
            holdings_df[col] = None
    
    # Select only the columns that exist and are required
    available_columns = [col for col in required_columns if col in holdings_df.columns]
    result_df = holdings_df[available_columns].copy()
    
    # Fill NaN values with appropriate defaults
    # Convert to object type first to allow None values
    if 'Sector' in result_df.columns:
        result_df['Sector'] = result_df['Sector'].astype('object')
        result_df.loc[result_df['Sector'].isna(), 'Sector'] = None
    if 'Country' in result_df.columns:
        result_df['Country'] = result_df['Country'].astype('object')
        result_df.loc[result_df['Country'].isna(), 'Country'] = None
    if 'Target_Allocation' in result_df.columns:
        result_df['Target_Allocation'] = result_df['Target_Allocation'].astype('object')
        result_df.loc[result_df['Target_Allocation'].isna(), 'Target_Allocation'] = None
    
    return result_df


def delete_transaction(transaction_id):
    """Delete a transaction and update holdings"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get transaction details before deleting
    cursor.execute("SELECT ticker, purchase_price, quantity FROM transactions WHERE id = ?", (transaction_id,))
    transaction = cursor.fetchone()
    
    if transaction:
        ticker, price, quantity = transaction
        
        # Delete transaction
        cursor.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
        
        # Update holdings (subtract the transaction)
        cursor.execute("SELECT total_shares, avg_price FROM holdings WHERE ticker = ?", (ticker,))
        holding = cursor.fetchone()
        
        if holding:
            old_shares, old_avg_price = holding
            new_shares = old_shares - quantity
            
            if new_shares <= 0:
                # Remove holding if no shares left
                cursor.execute("DELETE FROM holdings WHERE ticker = ?", (ticker,))
            else:
                # Recalculate average price (simplified - in reality this is complex)
                cursor.execute("""
                    UPDATE holdings 
                    SET total_shares = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE ticker = ?
                """, (new_shares, ticker))
    
    conn.commit()
    conn.close()


def update_holding_target_allocation(ticker, target_allocation):
    """Update target allocation for a holding"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE holdings 
        SET target_allocation = ?, last_updated = CURRENT_TIMESTAMP
        WHERE ticker = ?
    """, (target_allocation, ticker.upper()))
    
    conn.commit()
    conn.close()


def update_holding_price_data(ticker, current_price=None, annual_dividend=None):
    """Update current price and/or annual dividend for a holding"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Ensure columns exist
    try:
        cursor.execute("ALTER TABLE holdings ADD COLUMN current_price REAL")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute("ALTER TABLE holdings ADD COLUMN annual_dividend REAL")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    updates = []
    params = []
    
    # Handle current_price
    if current_price is not None:
        updates.append("current_price = ?")
        params.append(float(current_price))
    else:
        # If None, set to NULL in database (no parameter needed)
        updates.append("current_price = NULL")
    
    # Handle annual_dividend
    if annual_dividend is not None:
        updates.append("annual_dividend = ?")
        params.append(float(annual_dividend))
    else:
        # If None, set to NULL in database (no parameter needed)
        updates.append("annual_dividend = NULL")
    
    # Always update timestamp (no parameter needed)
    updates.append("last_updated = CURRENT_TIMESTAMP")
    
    # Add ticker as parameter for WHERE clause
    params.append(ticker.upper())
    
    # Build and execute query
    query = f"UPDATE holdings SET {', '.join(updates)} WHERE ticker = ?"
    cursor.execute(query, params)
    
    conn.commit()
    conn.close()


def get_manual_price_data():
    """Get all manually entered price data from holdings"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    
    # Get all holdings with price data (including NULL checks)
    df = pd.read_sql_query("""
        SELECT ticker, current_price, annual_dividend 
        FROM holdings 
        WHERE (current_price IS NOT NULL AND current_price > 0) 
           OR (annual_dividend IS NOT NULL AND annual_dividend > 0)
    """, conn)
    
    conn.close()
    return df


def delete_holding(ticker):
    """Delete all transactions for a specific ticker and remove from holdings"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    ticker_upper = ticker.upper()
    
    # Delete all transactions for this ticker
    cursor.execute("DELETE FROM transactions WHERE ticker = ?", (ticker_upper,))
    
    # Delete from holdings
    cursor.execute("DELETE FROM holdings WHERE ticker = ?", (ticker_upper,))
    
    conn.commit()
    conn.close()


def clear_all_data():
    """Clear all transactions and holdings (use with caution!)"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM transactions")
    cursor.execute("DELETE FROM holdings")
    cursor.execute("DELETE FROM dividends")
    
    conn.commit()
    conn.close()


# ==================== DIVIDEND FUNCTIONS ====================

def add_dividend(ticker, dividend_date, dividend_amount, currency):
    """
    Add a dividend payment for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        dividend_date: Date of dividend payment (YYYY-MM-DD)
        dividend_amount: Dividend amount per share
        currency: Currency of the dividend
        
    Returns:
        ID of the inserted dividend
    """
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    ticker_upper = ticker.upper()
    
    cursor.execute("""
        INSERT INTO dividends (ticker, dividend_date, dividend_amount, currency)
        VALUES (?, ?, ?, ?)
    """, (ticker_upper, dividend_date, float(dividend_amount), currency))
    
    dividend_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return dividend_id


def get_dividends_for_ticker(ticker):
    """
    Get all dividends for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with dividend data
    """
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    
    ticker_upper = ticker.upper()
    
    df = pd.read_sql_query("""
        SELECT id, ticker, dividend_date, dividend_amount, currency, created_at
        FROM dividends
        WHERE ticker = ?
        ORDER BY dividend_date DESC
    """, conn, params=(ticker_upper,))
    
    conn.close()
    return df


def get_all_dividends():
    """
    Get all dividends for all tickers.
    
    Returns:
        DataFrame with all dividend data
    """
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    
    df = pd.read_sql_query("""
        SELECT id, ticker, dividend_date, dividend_amount, currency, created_at
        FROM dividends
        ORDER BY ticker, dividend_date DESC
    """, conn)
    
    conn.close()
    return df


def delete_dividend(dividend_id):
    """
    Delete a specific dividend.
    
    Args:
        dividend_id: ID of the dividend to delete
    """
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM dividends WHERE id = ?", (dividend_id,))
    
    conn.commit()
    conn.close()


def delete_all_dividends_for_ticker(ticker):
    """
    Delete all dividends for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
    """
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    ticker_upper = ticker.upper()
    cursor.execute("DELETE FROM dividends WHERE ticker = ?", (ticker_upper,))
    
    conn.commit()
    conn.close()


def calculate_annual_dividend_from_dividends(ticker, year=None):
    """
    Calculate annual dividend from dividend records.
    If year is None, calculates for current year.
    
    Args:
        ticker: Stock ticker symbol
        year: Year to calculate for (None = current year)
        
    Returns:
        Total annual dividend per share for the year
    """
    from datetime import datetime
    
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    
    ticker_upper = ticker.upper()
    
    if year is None:
        year = datetime.now().year
    
    df = pd.read_sql_query("""
        SELECT SUM(dividend_amount) as total_dividend
        FROM dividends
        WHERE ticker = ? AND strftime('%Y', dividend_date) = ?
    """, conn, params=(ticker_upper, str(year)))
    
    conn.close()
    
    total = df.iloc[0]['total_dividend'] if not df.empty and pd.notna(df.iloc[0]['total_dividend']) else 0.0
    return float(total) if total else 0.0


