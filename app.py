"""
Stock Dividend Portfolio Tracker - Enhanced Version
A comprehensive Streamlit app with SQLite storage, transactions, and advanced analytics.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
import io

from data_fetcher import fetch_stock_data, fetch_currency_rates
from portfolio_calculator import (
    calculate_portfolio_metrics,
    calculate_dividend_yield,
    calculate_snowball_effect,
    calculate_advanced_statistics,
    calculate_yield_on_cost,
    calculate_rebalance_suggestions
)
from database import (
    init_database,
    add_transaction,
    get_all_transactions,
    get_all_holdings,
    get_holdings_as_portfolio_df,
    delete_transaction,
    delete_holding,
    update_holding_target_allocation,
    update_holding_price_data,
    get_manual_price_data,
    clear_all_data,
    add_dividend,
    get_dividends_for_ticker,
    get_all_dividends,
    delete_dividend,
    delete_all_dividends_for_ticker,
    calculate_annual_dividend_from_dividends
)
from utils import create_template_csv

# Page configuration
st.set_page_config(
    page_title="Stock Dividend Portfolio Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize database
init_database()

def main():
    """Main application function"""
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("📈 Portfolio Tracker")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["📊 Overview", "➕ Add Transaction", "✏️ Edit Prices", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Currency selection
        base_currency = st.selectbox(
            "Base Currency",
            ["SEK", "USD", "EUR"],
            index=0
        )
        
        st.markdown("---")
        
        # Quick stats
        holdings_df = get_all_holdings()
        if not holdings_df.empty:
            st.metric("Total Positions", len(holdings_df))
            total_shares = holdings_df['total_shares'].sum()
            st.metric("Total Shares", f"{total_shares:,.0f}")
        
        # Clear data button (with confirmation)
        st.markdown("---")
        if st.button("🗑️ Clear All Data", use_container_width=True, type="secondary"):
            if st.session_state.get('confirm_clear', False):
                clear_all_data()
                st.session_state.confirm_clear = False
                st.success("All data cleared!")
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm clearing all data")
    
    # Route to appropriate page
    if page == "📊 Overview":
        show_overview_page(base_currency)
    elif page == "➕ Add Transaction":
        show_add_transaction_page()
    elif page == "✏️ Edit Prices":
        show_edit_prices_page()
    elif page == "⚙️ Settings":
        show_settings_page()


def show_overview_page(base_currency):
    """Display the main overview dashboard"""
    
    st.title("📊 Portfolio Overview")
    st.markdown("---")
    
    # Get holdings from database
    portfolio_df = get_holdings_as_portfolio_df()
    
    if portfolio_df.empty:
        st.info("👆 No holdings found. Go to 'Add Transaction' to start building your portfolio.")
        return
    
    # Fetch currency rates
    with st.spinner("Fetching currency exchange rates..."):
        currency_rates = fetch_currency_rates()
    
    # Fetch stock data
    with st.spinner("Fetching stock data... This may take a moment."):
        portfolio_data = fetch_stock_data(portfolio_df, use_manual_prices=True)
    
    if portfolio_data is None or portfolio_data.empty:
        st.error("❌ Failed to fetch stock data. Please check your ticker symbols.")
        return
    
    # Check if prices were fetched successfully
    missing_prices = []
    if 'Current_Price' in portfolio_data.columns:
        missing_prices = portfolio_data[portfolio_data['Current_Price'] == 0]['Ticker'].tolist()
    
    manual_prices_used = []
    if 'Price_Source' in portfolio_data.columns:
        manual_prices_used = portfolio_data[portfolio_data['Price_Source'] == 'manual']['Ticker'].tolist()
    
    if manual_prices_used:
        st.success(f"✅ Använder manuellt angivna priser för: {', '.join(manual_prices_used)}")
    
    if missing_prices:
        with st.expander("🔍 Debug: Aktiekurser", expanded=False):
            st.write("**Kurser som saknas:**")
            for ticker in missing_prices:
                ticker_data = portfolio_data[portfolio_data['Ticker'] == ticker].iloc[0]
                current_price = ticker_data.get('Current_Price', 'N/A') if 'Current_Price' in ticker_data.index else 'N/A'
                price_source = ticker_data.get('Price_Source', 'N/A') if 'Price_Source' in ticker_data.index else 'N/A'
                st.write(f"- **{ticker}**: Current_Price = {current_price}, Price_Source = {price_source}")
            st.warning(f"⚠️ Aktiekurser kunde inte hämtas för: {', '.join(missing_prices)}. "
                      f"Gå till '✏️ Edit Prices' för att ange priser manuellt.")
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(
        portfolio_data,
        portfolio_df,
        base_currency,
        currency_rates
    )
    
    # Display key metrics
    display_key_metrics(metrics, base_currency, portfolio_data, portfolio_df, currency_rates)
    
    # Portfolio table with advanced metrics
    display_portfolio_table_enhanced(portfolio_data, portfolio_df, base_currency, currency_rates)
    
    # Rebalance suggestions
    if 'Target_Allocation' in portfolio_df.columns:
        display_rebalance_suggestions(portfolio_data, portfolio_df, base_currency, currency_rates)
    
    # Visualizations
    display_all_visualizations(portfolio_data, portfolio_df, base_currency, currency_rates)
    
    # Transaction history
    display_transaction_history()


def display_key_metrics(metrics, base_currency, portfolio_data, portfolio_df, currency_rates):
    """Display key portfolio metrics"""
    
    st.subheader("💰 Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Value",
            f"{metrics['total_value']:,.2f} {base_currency}"
        )
    
    with col2:
        st.metric(
            "Total Cost",
            f"{metrics['total_cost']:,.2f} {base_currency}"
        )
    
    with col3:
        st.metric(
            "Total Gain/Loss",
            f"{metrics['total_gain_loss']:,.2f} {base_currency}",
            delta=f"{metrics['gain_loss_percent']:.2f}%"
        )
    
    with col4:
        st.metric(
            "Annual Dividend Income",
            f"{metrics['annual_dividend_income']:,.2f} {base_currency}"
        )
    
    # Calculate additional metrics
    yoc_data = calculate_yield_on_cost(portfolio_data, portfolio_df, base_currency, currency_rates)
    avg_yoc = sum(yoc_data.values()) / len(yoc_data) if yoc_data else 0
    
    # Portfolio yield
    portfolio_yield = (metrics['annual_dividend_income'] / metrics['total_value'] * 100) if metrics['total_value'] > 0 else 0
    
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Portfolio Yield", f"{portfolio_yield:.2f}%")
    with col6:
        st.metric("Avg Yield on Cost", f"{avg_yoc:.2f}%" if avg_yoc > 0 else "N/A")


def display_portfolio_table_enhanced(portfolio_data, portfolio_df, base_currency, currency_rates):
    """Display enhanced portfolio table with all metrics"""
    
    st.subheader("📋 Portfolio Holdings")
    
    # Merge data
    # Use suffixes to avoid column name conflicts
    # portfolio_df has user's Currency, portfolio_data has yfinance Currency
    display_df = portfolio_df.merge(portfolio_data, left_on='Ticker', right_on='Ticker', how='left', suffixes=('', '_yf'))
    
    # Debug: Check if Current_Price was fetched
    if 'Current_Price' in display_df.columns:
        zero_prices = display_df[display_df['Current_Price'] == 0]['Ticker'].tolist()
        if zero_prices:
            st.warning(f"⚠️ Aktiekurser saknas för: {', '.join(zero_prices)}")
        
        # Debug: Show dividend yield calculation details
        with st.expander("🔍 Debug: Dividend Yield Beräkning"):
            debug_cols = ['Ticker', 'Current_Price_Base', 'Annual_Dividend_Base', 'Dividend_Yield', 'Shares']
            if all(col in display_df.columns for col in debug_cols):
                debug_df = display_df[debug_cols].copy()
                debug_df['Total_Dividend'] = debug_df['Shares'] * debug_df['Annual_Dividend_Base']
                debug_df['Formula'] = debug_df.apply(
                    lambda row: f"({row['Annual_Dividend_Base']:.2f} / {row['Current_Price_Base']:.2f}) × 100" 
                    if row['Current_Price_Base'] > 0 else "N/A (pris = 0)",
                    axis=1
                )
                st.dataframe(debug_df, use_container_width=True)
                st.caption("💡 **Förklaring**: Dividend Yield = (Årlig Utdelning per Aktie / Aktuellt Pris) × 100")
                st.caption("⚠️ **Viktigt**: 'Annual_Dividend_Base' ska vara per aktie, inte total utdelning!")
    
    # Keep both currencies: Currency (user's) and Currency_yf (yfinance's)
    # We need Currency_yf to convert Current_Price correctly
    
    # Ensure required columns exist with defaults
    if 'Sector' not in display_df.columns:
        display_df['Sector'] = None
    if 'Country' not in display_df.columns:
        display_df['Country'] = None
    if 'Target_Allocation' not in display_df.columns:
        display_df['Target_Allocation'] = None
    if 'Currency' not in display_df.columns:
        display_df['Currency'] = 'USD'  # Default currency (shouldn't happen if data is correct)
    if 'Shares' not in display_df.columns:
        display_df['Shares'] = 0
    if 'Avg_Price' not in display_df.columns:
        display_df['Avg_Price'] = 0
    if 'Current_Price' not in display_df.columns:
        display_df['Current_Price'] = 0
    
    # CRITICAL: Ensure we use Currency from portfolio_df (user's input), not from yfinance
    # After merge, Currency should come from portfolio_df, but let's be explicit
    if 'Currency' in portfolio_df.columns:
        # Map Currency from original portfolio_df to ensure correct values
        currency_map = dict(zip(portfolio_df['Ticker'], portfolio_df['Currency']))
        display_df['Currency'] = display_df['Ticker'].map(currency_map).fillna(display_df.get('Currency', 'USD'))
    
    # Calculate values with safe access
    # IMPORTANT: Current_Price from yfinance is in yfinance's currency (Currency_yf)
    # Avg_Price from user is in user's currency (Currency)
    
    # Get yfinance currency (fallback to user's currency if not available)
    if 'Currency_yf' in display_df.columns:
        display_df['YF_Currency'] = display_df['Currency_yf'].fillna(display_df.get('Currency', 'USD'))
    else:
        # If no yfinance currency, assume it's the same as user's currency
        display_df['YF_Currency'] = display_df.get('Currency', 'USD')
    
    # Convert Current_Price from yfinance currency to base currency
    # If yfinance currency differs from user's currency, convert through user's currency first
    display_df['Current_Price_Base'] = display_df.apply(
        lambda row: (
            float(row.get('Current_Price', 0)) * 
            float(currency_rates.get(f"{row.get('YF_Currency', 'USD')}/{base_currency}", 1.0))
            if pd.notna(row.get('Current_Price')) and float(row.get('Current_Price', 0)) > 0
            else 0.0
        ),
        axis=1
    )
    
    # Avg_Price is already in user's currency, convert to base currency
    display_df['Avg_Price_Base'] = display_df.apply(
        lambda row: row.get('Avg_Price', 0) * currency_rates.get(f"{row.get('Currency', 'USD')}/{base_currency}", 1.0),
        axis=1
    )
    
    display_df['Market_Value'] = display_df['Shares'] * display_df['Current_Price_Base']
    display_df['Cost_Basis'] = display_df['Shares'] * display_df['Avg_Price_Base']
    display_df['Gain_Loss'] = display_df['Market_Value'] - display_df['Cost_Basis']
    
    # Calculate percentage, handle division by zero
    display_df['Gain_Loss_Pct'] = display_df.apply(
        lambda row: round((row['Gain_Loss'] / row['Cost_Basis'] * 100), 2) if row['Cost_Basis'] > 0 else 0.0,
        axis=1
    )
    
    # Calculate dividend metrics
    display_df['Annual_Dividend_Base'] = display_df.apply(
        lambda row: row.get('Annual_Dividend', 0) * currency_rates.get(f"{row.get('Currency', 'USD')}/{base_currency}", 1.0) if pd.notna(row.get('Annual_Dividend')) else 0,
        axis=1
    )
    
    # Calculate Dividend Yield, handle division by zero and NaN
    # Dividend Yield = (Annual Dividend per Share / Current Price) × 100
    # Note: Annual_Dividend should be per share, not total
    display_df['Dividend_Yield'] = display_df.apply(
        lambda row: round((row['Annual_Dividend_Base'] / row['Current_Price_Base'] * 100), 2) 
        if pd.notna(row['Current_Price_Base']) and row['Current_Price_Base'] > 0 and pd.notna(row['Annual_Dividend_Base']) and row['Annual_Dividend_Base'] > 0
        else 0.0,
        axis=1
    )
    
    # Calculate Yield on Cost, handle division by zero and NaN
    display_df['Yield_on_Cost'] = display_df.apply(
        lambda row: round((row['Annual_Dividend_Base'] / row['Avg_Price_Base'] * 100), 2) 
        if pd.notna(row['Avg_Price_Base']) and row['Avg_Price_Base'] > 0 and pd.notna(row['Annual_Dividend_Base']) 
        else 0.0,
        axis=1
    )
    
    display_df['Dividend_Income'] = display_df['Shares'] * display_df['Annual_Dividend_Base']
    
    # Payout ratio
    display_df['Payout_Ratio'] = display_df.apply(
        lambda row: row.get('Payout_Ratio', 0) * 100 if pd.notna(row.get('Payout_Ratio')) else 0,
        axis=1
    )
    
    # Select and format columns
    display_columns = [
        'Ticker', 'Sector', 'Country', 'Shares', 
        f'Current Price ({base_currency})', f'Avg Price ({base_currency})',
        'Market Value', 'Cost Basis', 'Gain/Loss', 'Gain/Loss %',
        'Dividend Yield %', 'Yield on Cost %', 'Payout Ratio %', 'Annual Dividend Income'
    ]
    
    formatted_df = pd.DataFrame()
    formatted_df['Ticker'] = display_df['Ticker']
    formatted_df['Sector'] = display_df['Sector'].fillna('N/A') if 'Sector' in display_df.columns else 'N/A'
    formatted_df['Country'] = display_df['Country'].fillna('N/A') if 'Country' in display_df.columns else 'N/A'
    formatted_df['Shares'] = display_df['Shares'].apply(lambda x: f"{x:,.2f}")
    formatted_df[f'Current Price ({base_currency})'] = display_df['Current_Price_Base'].apply(
        lambda x: f"{float(x):,.2f}" if pd.notna(x) and float(x) > 0 else "N/A"
    )
    formatted_df[f'Avg Price ({base_currency})'] = display_df['Avg_Price_Base'].apply(lambda x: f"{x:,.2f}")
    formatted_df['Market Value'] = display_df['Market_Value'].apply(
        lambda x: f"{x:,.2f}" if pd.notna(x) and x != 0 else "N/A"
    )
    formatted_df['Cost Basis'] = display_df['Cost_Basis'].apply(lambda x: f"{x:,.2f}")
    formatted_df['Gain/Loss'] = display_df['Gain_Loss'].apply(lambda x: f"{x:,.2f}")
    formatted_df['Gain/Loss %'] = display_df['Gain_Loss_Pct'].apply(lambda x: f"{x:,.2f}%")
    formatted_df['Dividend Yield %'] = display_df['Dividend_Yield'].apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) and x != 0 else "N/A"
    )
    formatted_df['Yield on Cost %'] = display_df['Yield_on_Cost'].apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) and x != 0 else "N/A"
    )
    formatted_df['Payout Ratio %'] = display_df['Payout_Ratio'].apply(lambda x: f"{x:.2f}%" if x > 0 else "N/A")
    formatted_df['Annual Dividend Income'] = display_df['Dividend_Income'].apply(lambda x: f"{x:,.2f}")
    
    st.dataframe(formatted_df, use_container_width=True, hide_index=True)
    
    # Add section to delete individual holdings
    st.markdown("---")
    with st.expander("🗑️ Ta bort innehav", expanded=False):
        st.markdown("**Varning:** Detta tar bort alla transaktioner för det valda innehavet.")
        
        if not portfolio_df.empty:
            ticker_list = portfolio_df['Ticker'].tolist()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_ticker = st.selectbox(
                    "Välj innehav att ta bort",
                    ticker_list,
                    key="delete_holding_select",
                    label_visibility="collapsed"
                )
            
            with col2:
                if st.button("🗑️ Ta bort", use_container_width=True, type="secondary", key="delete_holding_button"):
                    if st.session_state.get(f'confirm_delete_{selected_ticker}', False):
                        try:
                            delete_holding(selected_ticker)
                            st.success(f"✅ {selected_ticker} har tagits bort!")
                            st.session_state[f'confirm_delete_{selected_ticker}'] = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Fel vid borttagning: {str(e)}")
                    else:
                        st.session_state[f'confirm_delete_{selected_ticker}'] = True
                        st.warning(f"Klicka igen för att bekräfta borttagning av {selected_ticker}")
        else:
            st.info("Inga innehav att ta bort.")


def display_rebalance_suggestions(portfolio_data, portfolio_df, base_currency, currency_rates):
    """Display rebalancing suggestions"""
    
    st.subheader("⚖️ Rebalance Suggestions")
    
    suggestions_df = calculate_rebalance_suggestions(portfolio_data, portfolio_df, base_currency, currency_rates)
    
    if not suggestions_df.empty:
        # Format the dataframe
        display_suggestions = suggestions_df.copy()
        for col in ['Current_Value', 'Target_Value', 'Value_Difference']:
            if col in display_suggestions.columns:
                display_suggestions[col] = display_suggestions[col].apply(
                    lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else str(x)
                )
        
        st.dataframe(display_suggestions, use_container_width=True, hide_index=True)
    else:
        st.info("No target allocations set. Set target allocations in Settings to see rebalancing suggestions.")


def display_all_visualizations(portfolio_data, portfolio_df, base_currency, currency_rates):
    """Display all visualizations"""
    
    st.subheader("📈 Visualizations")
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Prepare data
        # Use suffixes to avoid conflicts, prioritize user's currency
        chart_df = portfolio_df.merge(portfolio_data, left_on='Ticker', right_on='Ticker', how='left', suffixes=('', '_yf'))
        
        # If Currency was duplicated, use the one from portfolio_df (user's input)
        if 'Currency_yf' in chart_df.columns:
            chart_df = chart_df.drop(columns=['Currency_yf'])
        
        # Ensure required columns exist with defaults
        if 'Currency' not in chart_df.columns:
            chart_df['Currency'] = 'USD'  # Default currency
        if 'Shares' not in chart_df.columns:
            chart_df['Shares'] = 0
        if 'Current_Price' not in chart_df.columns:
            chart_df['Current_Price'] = 0
        
        # CRITICAL: Ensure we use Currency from portfolio_df (user's input), not from yfinance
        if 'Currency' in portfolio_df.columns:
            currency_map = dict(zip(portfolio_df['Ticker'], portfolio_df['Currency']))
            chart_df['Currency'] = chart_df['Ticker'].map(currency_map).fillna(chart_df.get('Currency', 'USD'))
        
        chart_df['Market_Value'] = chart_df.apply(
            lambda row: row.get('Shares', 0) * row.get('Current_Price', 0) * 
            currency_rates.get(f"{row.get('Currency', 'USD')}/{base_currency}", 1.0),
            axis=1
        )
        
        # Tab layout for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Allocation", "Dividend Yield", "Snowball Effect", "Calendar Year", "Sector/Country"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                # Portfolio allocation by value
                fig_pie = px.pie(
                    chart_df,
                    values='Market_Value',
                    names='Ticker',
                    title='Portfolio Allocation by Value'
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Dividend income allocation
                chart_df['Dividend_Income'] = chart_df.apply(
                    lambda row: row.get('Shares', 0) * row.get('Annual_Dividend', 0) * 
                    currency_rates.get(f"{row.get('Currency', 'USD')}/{base_currency}", 1.0) if pd.notna(row.get('Annual_Dividend')) else 0,
                    axis=1
                )
                fig_pie2 = px.pie(
                    chart_df,
                    values='Dividend_Income',
                    names='Ticker',
                    title='Dividend Income Allocation'
                )
                fig_pie2.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie2, use_container_width=True)
        
        with tab2:
            # Dividend yield bar chart
            chart_df['Dividend_Yield'] = chart_df.apply(
                lambda row: calculate_dividend_yield(
                    row.get('Annual_Dividend', 0),
                    row.get('Current_Price', 0),
                    row.get('Currency', 'USD'),
                    base_currency,
                    currency_rates
                ) if pd.notna(row.get('Annual_Dividend')) else 0,
                axis=1
            )
            
            fig_bar = px.bar(
                chart_df,
                x='Ticker',
                y='Dividend_Yield',
                title='Dividend Yield by Stock',
                labels={'Dividend_Yield': 'Dividend Yield (%)'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab3:
            # Snowball effect
            st.markdown("**Snöbollseffekt: Portföljutveckling med återinvesterade utdelningar**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                years = st.slider("Antal år", 1, 30, 10, 1, key="snowball_years")
            with col2:
                reinvest = st.checkbox("Återinvestera utdelningar", value=True, key="snowball_reinvest")
            with col3:
                growth_rate = st.slider("Årlig tillväxt (%)", 0.0, 15.0, 5.0, 0.5, key="snowball_growth") / 100
            
            projections = calculate_snowball_effect(
                portfolio_data,
                portfolio_df,
                base_currency,
                currency_rates,
                years=years,
                reinvest_dividends=reinvest,
                annual_growth_rate=growth_rate
            )
            
            if not projections.empty:
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Portföljvärde över tid', 'Årlig utdelningsinkomst över tid'),
                    vertical_spacing=0.15
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=projections['Year'],
                        y=projections['Portfolio_Value'],
                        mode='lines+markers',
                        name='Portföljvärde',
                        line=dict(color='#1f77b4', width=3)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=projections['Year'],
                        y=projections['Annual_Dividend_Income'],
                        mode='lines+markers',
                        name='Årlig utdelning',
                        line=dict(color='#2ca02c', width=3)
                    ),
                    row=2, col=1
                )
                
                fig.update_xaxes(title_text="År", row=2, col=1)
                fig.update_yaxes(title_text=f"Värde ({base_currency})", row=1, col=1)
                fig.update_yaxes(title_text=f"Utdelning ({base_currency})", row=2, col=1)
                fig.update_layout(height=700, showlegend=True)
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Calendar year view using actual dividend dates
            st.markdown("**Kalenderår: Utdelningar per månad**")
            
            # Get all dividends for current year
            current_year = datetime.now().year
            all_dividends_df = get_all_dividends()
            
            if not all_dividends_df.empty:
                # Filter for current year
                all_dividends_df['dividend_date'] = pd.to_datetime(all_dividends_df['dividend_date'])
                current_year_dividends = all_dividends_df[all_dividends_df['dividend_date'].dt.year == current_year].copy()
                
                if not current_year_dividends.empty:
                    # Get shares for each ticker
                    shares_map = dict(zip(portfolio_df['Ticker'], portfolio_df['Shares']))
                    
                    # Calculate total dividend per month (considering shares)
                    monthly_data = []
                    for _, div_row in current_year_dividends.iterrows():
                        ticker = div_row['ticker']
                        shares = shares_map.get(ticker, 0)
                        dividend_per_share = div_row['dividend_amount']
                        div_currency = div_row['currency']
                        
                        # Convert to base currency
                        rate = currency_rates.get(f"{div_currency}/{base_currency}", 1.0)
                        total_dividend = shares * dividend_per_share * rate
                        
                        month = div_row['dividend_date'].month
                        monthly_data.append({
                            'Month': month,
                            'Ticker': ticker,
                            'Dividend': total_dividend,
                            'Date': div_row['dividend_date'].strftime('%Y-%m-%d')
                        })
                    
                    if monthly_data:
                        monthly_df = pd.DataFrame(monthly_data)
                        monthly_totals = monthly_df.groupby('Month')['Dividend'].sum().reset_index()
                        monthly_totals['Month_Name'] = monthly_totals['Month'].map({
                            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                        })
                        
                        fig_bar = px.bar(
                            monthly_totals,
                            x='Month_Name',
                            y='Dividend',
                            title=f'Faktiska Utdelningar per Månad {current_year}',
                            labels={'Dividend': f'Utdelning ({base_currency})', 'Month_Name': 'Månad'},
                            color='Dividend',
                            color_continuous_scale='Greens'
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Show detailed table
                        with st.expander("📋 Detaljerad Utdelningslista"):
                            display_monthly = monthly_df.copy()
                            display_monthly = display_monthly.sort_values(['Month', 'Ticker'])
                            display_monthly.columns = ['Månad', 'Ticker', 'Utdelning', 'Datum']
                            display_monthly['Månad'] = display_monthly['Månad'].map({
                                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                            })
                            st.dataframe(display_monthly, use_container_width=True, hide_index=True)
                    else:
                        st.info(f"Inga utdelningar registrerade för {current_year}")
                else:
                    st.info(f"Inga utdelningar registrerade för {current_year}. Lägg till utdelningar med datum i '✏️ Redigera Priser & Utdelningar'.")
            else:
                st.info("Inga utdelningar registrerade. Lägg till utdelningar med datum i '✏️ Redigera Priser & Utdelningar'.")
        
        with tab5:
            col1, col2 = st.columns(2)
            
            with col1:
                # Sector distribution
                if 'Sector' in chart_df.columns and chart_df['Sector'].notna().any():
                    sector_totals = chart_df.groupby('Sector')['Market_Value'].sum().reset_index()
                    fig_sector = px.pie(
                        sector_totals,
                        values='Market_Value',
                        names='Sector',
                        title='Sector Distribution'
                    )
                    st.plotly_chart(fig_sector, use_container_width=True)
                else:
                    st.info("No sector data available. Add sector information when adding transactions.")
            
            with col2:
                # Geographic distribution
                if 'Country' in chart_df.columns and chart_df['Country'].notna().any():
                    country_totals = chart_df.groupby('Country')['Market_Value'].sum().reset_index()
                    fig_country = px.pie(
                        country_totals,
                        values='Market_Value',
                        names='Country',
                        title='Geographic Exposure'
                    )
                    st.plotly_chart(fig_country, use_container_width=True)
                else:
                    st.info("No country data available. Add country information when adding transactions.")
    
    except ImportError:
        st.warning("Plotly is required for visualizations. Install with: pip install plotly")


def display_transaction_history():
    """Display transaction history with option to delete individual transactions"""
    
    st.subheader("📜 Transaction History")
    
    transactions_df = get_all_transactions()
    
    if transactions_df.empty:
        st.info("No transactions recorded yet.")
        return
    
    # Format for display
    display_df = transactions_df.copy()
    if 'purchase_date' in display_df.columns:
        display_df['purchase_date'] = pd.to_datetime(display_df['purchase_date']).dt.strftime('%Y-%m-%d')
    
    # Select columns to display
    display_columns = ['id', 'ticker', 'purchase_date', 'purchase_price', 'quantity', 'currency', 'sector', 'country', 'target_allocation']
    available_columns = [col for col in display_columns if col in display_df.columns]
    
    st.dataframe(display_df[available_columns], use_container_width=True, hide_index=True)
    
    # Add option to delete individual transactions
    st.markdown("---")
    with st.expander("🗑️ Ta bort transaktion", expanded=False):
        st.markdown("**Varning:** Detta tar bort transaktionen och uppdaterar innehavet automatiskt.")
        
        if not transactions_df.empty:
            # Create a readable list of transactions
            transaction_options = []
            for idx, row in transactions_df.iterrows():
                trans_id = row.get('id', '')
                ticker = row.get('ticker', '')
                date = row.get('purchase_date', '')
                qty = row.get('quantity', 0)
                price = row.get('purchase_price', 0)
                transaction_options.append(f"ID {trans_id}: {ticker} - {qty} @ {price} ({date})")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_transaction = st.selectbox(
                    "Välj transaktion att ta bort",
                    transaction_options,
                    key="delete_transaction_select",
                    label_visibility="collapsed"
                )
            
            with col2:
                if st.button("🗑️ Ta bort", use_container_width=True, type="secondary", key="delete_transaction_button"):
                    # Extract transaction ID from selection
                    trans_id = int(selected_transaction.split("ID ")[1].split(":")[0])
                    
                    if st.session_state.get(f'confirm_delete_trans_{trans_id}', False):
                        try:
                            delete_transaction(trans_id)
                            st.success(f"✅ Transaktion {trans_id} har tagits bort!")
                            st.session_state[f'confirm_delete_trans_{trans_id}'] = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Fel vid borttagning: {str(e)}")
                    else:
                        st.session_state[f'confirm_delete_trans_{trans_id}'] = True
                        st.warning(f"Klicka igen för att bekräfta borttagning av transaktion {trans_id}")


def show_add_transaction_page():
    """Display the add transaction form"""
    
    st.title("➕ Add Transaction")
    st.markdown("---")
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.text_input("Ticker Symbol *", placeholder="AAPL or investor-b.st", help="Enter stock ticker (e.g., AAPL, investor-b.st)")
            purchase_date = st.date_input("Purchase Date *", value=date.today())
            purchase_price = st.number_input("Purchase Price *", min_value=0.0, step=0.01, format="%.2f")
            quantity = st.number_input("Quantity (Shares) *", min_value=0.0, step=0.01, format="%.2f")
        
        with col2:
            currency = st.selectbox("Currency *", ["SEK", "USD", "EUR"], index=0)
            sector = st.text_input("Sector", placeholder="Technology, Finance, etc.")
            country = st.text_input("Country", placeholder="USA, Sweden, etc.")
            target_allocation = st.number_input("Target Allocation %", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f")
            if target_allocation == 0:
                target_allocation = None
        
        notes = st.text_area("Notes (optional)", placeholder="Any additional notes about this transaction")
        
        submitted = st.form_submit_button("💾 Add Transaction", use_container_width=True)
        
        if submitted:
            if not ticker or purchase_price <= 0 or quantity <= 0:
                st.error("Please fill in all required fields (marked with *)")
            else:
                try:
                    # Ensure database is initialized
                    init_database()
                    
                    transaction_id = add_transaction(
                        ticker=ticker.upper(),
                        purchase_date=purchase_date,
                        purchase_price=purchase_price,
                        quantity=quantity,
                        currency=currency,
                        sector=sector if sector else None,
                        country=country if country else None,
                        target_allocation=target_allocation,
                        notes=notes if notes else None
                    )
                    st.success(f"✅ Transaction added successfully! (ID: {transaction_id})")
                    st.balloons()
                    st.rerun()  # Refresh to show new transaction
                except Exception as e:
                    st.error(f"❌ Error adding transaction: {str(e)}")
                    st.exception(e)  # Show full traceback for debugging


def show_settings_page():
    """Display settings page"""
    
    st.title("⚙️ Settings")
    st.markdown("---")
    
    # Get current holdings
    holdings_df = get_all_holdings()
    
    if holdings_df.empty:
        st.info("No holdings to configure. Add transactions first.")
        return
    
    st.subheader("Target Allocations")
    st.markdown("Set target allocation percentages for each holding to enable rebalancing suggestions.")
    
    # Create form for updating target allocations
    with st.form("target_allocation_form"):
        allocation_data = []
        
        for idx, row in holdings_df.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{row['ticker']}**")
            with col2:
                current_alloc = row.get('target_allocation', 0) or 0
                new_alloc = st.number_input(
                    f"Target %",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(current_alloc),
                    step=0.1,
                    format="%.1f",
                    key=f"alloc_{row['ticker']}"
                )
                allocation_data.append({
                    'ticker': row['ticker'],
                    'allocation': new_alloc if new_alloc > 0 else None
                })
        
        submitted = st.form_submit_button("💾 Save Target Allocations", use_container_width=True)
        
        if submitted:
            for item in allocation_data:
                update_holding_target_allocation(item['ticker'], item['allocation'])
            st.success("✅ Target allocations updated!")
            st.rerun()
    
    st.markdown("---")
    st.subheader("Data Management")
    
    # Export data
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export Holdings to CSV", use_container_width=True):
            csv = holdings_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"holdings_export_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📥 Export Transactions to CSV", use_container_width=True):
            transactions_df = get_all_transactions()
            csv = transactions_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"transactions_export_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


def show_edit_prices_page():
    """Display page for manually editing prices and dividend data"""
    
    st.title("✏️ Redigera Priser & Utdelningar")
    st.markdown("---")
    st.markdown("**Ange aktuella priser och utdelningar manuellt. Dessa värden används istället för automatisk hämtning från Yahoo Finance.**")
    
    # Get current holdings
    holdings_df = get_all_holdings()
    
    if holdings_df.empty:
        st.info("Inga innehav att redigera. Lägg till transaktioner först.")
        return
    
    # Create tabs for Prices and Dividends
    tab1, tab2 = st.tabs(["💰 Priser", "📅 Utdelningar med Datum"])
    
    with tab1:
        st.subheader("Manuell Inmatning av Priser")
        
        with st.form("edit_prices_form"):
            price_data = []
            
            for idx, row in holdings_df.iterrows():
                with st.expander(f"**{row['ticker']}**", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        current_price = row.get('current_price', 0) or 0
                        new_price = st.number_input(
                            f"Aktuellt Pris ({row.get('currency', 'USD')})",
                            min_value=0.0,
                            value=float(current_price) if current_price else 0.0,
                            step=0.01,
                            format="%.2f",
                            key=f"price_{row['ticker']}",
                            help="Lämna tomt (0) för att använda automatisk hämtning från Yahoo Finance"
                        )
                    
                    with col2:
                        # Show calculated annual dividend from dividend records
                        annual_from_dividends = calculate_annual_dividend_from_dividends(row['ticker'])
                        annual_dividend = row.get('annual_dividend', 0) or 0
                        
                        if annual_from_dividends > 0:
                            st.info(f"📊 Beräknad årlig utdelning från utdelningsregister: **{annual_from_dividends:.2f} {row.get('currency', 'USD')}**")
                        
                        new_dividend = st.number_input(
                            f"Årlig Utdelning (Fallback) ({row.get('currency', 'USD')})",
                            min_value=0.0,
                            value=float(annual_dividend) if annual_dividend else 0.0,
                            step=0.01,
                            format="%.2f",
                            key=f"dividend_{row['ticker']}",
                            help="Används endast om inga utdelningar med datum är registrerade"
                        )
                    
                    price_data.append({
                        'ticker': row['ticker'],
                        'current_price': new_price if new_price > 0 else None,
                        'annual_dividend': new_dividend if new_dividend > 0 else None
                    })
            
            submitted = st.form_submit_button("💾 Spara Priser", use_container_width=True)
            
            if submitted:
                try:
                    for item in price_data:
                        update_holding_price_data(
                            item['ticker'],
                            item['current_price'],
                            item['annual_dividend']
                        )
                    
                    # Clear cache to ensure new prices are loaded
                    fetch_stock_data.clear()
                    
                    st.success("✅ Priser uppdaterade!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Fel vid sparning: {str(e)}")
                    st.exception(e)
        
        st.markdown("---")
        st.info("💡 **Tips:** Om du lämnar pris som 0, kommer appen att försöka hämta data automatiskt från Yahoo Finance.")
    
    with tab2:
        st.subheader("Hantera Utdelningar med Datum")
        st.markdown("**Lägg till flera utdelningar per år med specifika datum. Appen beräknar automatiskt total årlig utdelning.**")
        
        # Select ticker
        ticker_options = [f"{row['ticker']} ({row.get('currency', 'USD')})" for _, row in holdings_df.iterrows()]
        selected_ticker_str = st.selectbox(
            "Välj Ticker",
            ticker_options,
            key="dividend_ticker_select"
        )
        
        if selected_ticker_str:
            selected_ticker = selected_ticker_str.split(" (")[0]
            selected_currency = holdings_df[holdings_df['ticker'] == selected_ticker].iloc[0].get('currency', 'USD')
            
            # Show existing dividends
            st.markdown("### 📋 Befintliga Utdelningar")
            dividends_df = get_dividends_for_ticker(selected_ticker)
            
            if not dividends_df.empty:
                # Format for display
                display_dividends = dividends_df[['dividend_date', 'dividend_amount', 'currency']].copy()
                display_dividends.columns = ['Datum', 'Belopp per Aktie', 'Valuta']
                display_dividends['Årlig Totalt'] = display_dividends['Belopp per Aktie'].sum()
                
                st.dataframe(display_dividends, use_container_width=True, hide_index=True)
                
                # Calculate annual total
                current_year = datetime.now().year
                year_total = calculate_annual_dividend_from_dividends(selected_ticker, current_year)
                st.success(f"📊 **Total utdelning {current_year}: {year_total:.2f} {selected_currency} per aktie**")
                
                # Delete dividend option
                st.markdown("#### 🗑️ Ta bort utdelning")
                dividend_options = [
                    f"ID {row['id']}: {row['dividend_date']} - {row['dividend_amount']:.2f} {row['currency']}"
                    for _, row in dividends_df.iterrows()
                ]
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_dividend = st.selectbox(
                        "Välj utdelning att ta bort",
                        dividend_options,
                        key="delete_dividend_select"
                    )
                
                with col2:
                    if st.button("🗑️ Ta bort", use_container_width=True, key="delete_dividend_button"):
                        dividend_id = int(selected_dividend.split("ID ")[1].split(":")[0])
                        try:
                            delete_dividend(dividend_id)
                            st.success("✅ Utdelning borttagen!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Fel: {str(e)}")
            else:
                st.info("Inga utdelningar registrerade för denna ticker.")
            
            st.markdown("---")
            
            # Add new dividend
            st.markdown("### ➕ Lägg till Ny Utdelning")
            
            with st.form("add_dividend_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    dividend_date = st.date_input(
                        "Utdelningsdatum *",
                        value=date.today(),
                        key="dividend_date_input"
                    )
                
                with col2:
                    dividend_amount = st.number_input(
                        f"Utdelning per Aktie ({selected_currency}) *",
                        min_value=0.0,
                        step=0.01,
                        format="%.2f",
                        key="dividend_amount_input",
                        help="Utdelning per aktie, inte total utdelning"
                    )
                
                submitted = st.form_submit_button("💾 Lägg till Utdelning", use_container_width=True)
                
                if submitted:
                    if dividend_amount <= 0:
                        st.error("Utdelning måste vara större än 0")
                    else:
                        try:
                            add_dividend(
                                ticker=selected_ticker,
                                dividend_date=dividend_date,
                                dividend_amount=dividend_amount,
                                currency=selected_currency
                            )
                            st.success(f"✅ Utdelning tillagd: {dividend_amount:.2f} {selected_currency} per aktie den {dividend_date}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Fel vid tillägg: {str(e)}")
                            st.exception(e)


if __name__ == "__main__":
    main()

