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
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
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
    
    # Rename columns to match expected format
    holdings_df = holdings_df.rename(columns={
        'total_shares': 'Shares',
        'avg_price': 'Avg_Price',
        'target_allocation': 'Target_Allocation'
    })
    
    return holdings_df[['Ticker', 'Shares', 'Avg_Price', 'Currency', 'Sector', 'Country', 'Target_Allocation']]


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


def clear_all_data():
    """Clear all transactions and holdings (use with caution!)"""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM transactions")
    cursor.execute("DELETE FROM holdings")
    
    conn.commit()
    conn.close()

