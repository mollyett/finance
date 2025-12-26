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
    update_holding_target_allocation,
    clear_all_data
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
            ["📊 Overview", "➕ Add Transaction", "⚙️ Settings"],
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
        portfolio_data = fetch_stock_data(portfolio_df)
    
    if portfolio_data is None or portfolio_data.empty:
        st.error("❌ Failed to fetch stock data. Please check your ticker symbols.")
        return
    
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
    display_df = portfolio_df.merge(portfolio_data, left_on='Ticker', right_on='Ticker', how='left')
    
    # Calculate values
    display_df['Current_Price_Base'] = display_df.apply(
        lambda row: row['Current_Price'] * currency_rates.get(f"{row['Currency']}/{base_currency}", 1.0),
        axis=1
    )
    display_df['Avg_Price_Base'] = display_df.apply(
        lambda row: row['Avg_Price'] * currency_rates.get(f"{row['Currency']}/{base_currency}", 1.0),
        axis=1
    )
    display_df['Market_Value'] = display_df['Shares'] * display_df['Current_Price_Base']
    display_df['Cost_Basis'] = display_df['Shares'] * display_df['Avg_Price_Base']
    display_df['Gain_Loss'] = display_df['Market_Value'] - display_df['Cost_Basis']
    display_df['Gain_Loss_Pct'] = (display_df['Gain_Loss'] / display_df['Cost_Basis'] * 100).round(2)
    
    # Calculate dividend metrics
    display_df['Annual_Dividend_Base'] = display_df.apply(
        lambda row: row.get('Annual_Dividend', 0) * currency_rates.get(f"{row['Currency']}/{base_currency}", 1.0) if pd.notna(row.get('Annual_Dividend')) else 0,
        axis=1
    )
    display_df['Dividend_Yield'] = (display_df['Annual_Dividend_Base'] / display_df['Current_Price_Base'] * 100).round(2)
    display_df['Yield_on_Cost'] = (display_df['Annual_Dividend_Base'] / display_df['Avg_Price_Base'] * 100).round(2)
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
    formatted_df['Sector'] = display_df.get('Sector', 'N/A')
    formatted_df['Country'] = display_df.get('Country', 'N/A')
    formatted_df['Shares'] = display_df['Shares'].apply(lambda x: f"{x:,.2f}")
    formatted_df[f'Current Price ({base_currency})'] = display_df['Current_Price_Base'].apply(lambda x: f"{x:,.2f}")
    formatted_df[f'Avg Price ({base_currency})'] = display_df['Avg_Price_Base'].apply(lambda x: f"{x:,.2f}")
    formatted_df['Market Value'] = display_df['Market_Value'].apply(lambda x: f"{x:,.2f}")
    formatted_df['Cost Basis'] = display_df['Cost_Basis'].apply(lambda x: f"{x:,.2f}")
    formatted_df['Gain/Loss'] = display_df['Gain_Loss'].apply(lambda x: f"{x:,.2f}")
    formatted_df['Gain/Loss %'] = display_df['Gain_Loss_Pct'].apply(lambda x: f"{x:,.2f}%")
    formatted_df['Dividend Yield %'] = display_df['Dividend_Yield'].apply(lambda x: f"{x:.2f}%")
    formatted_df['Yield on Cost %'] = display_df['Yield_on_Cost'].apply(lambda x: f"{x:.2f}%")
    formatted_df['Payout Ratio %'] = display_df['Payout_Ratio'].apply(lambda x: f"{x:.2f}%" if x > 0 else "N/A")
    formatted_df['Annual Dividend Income'] = display_df['Dividend_Income'].apply(lambda x: f"{x:,.2f}")
    
    st.dataframe(formatted_df, use_container_width=True, hide_index=True)


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
        chart_df = portfolio_df.merge(portfolio_data, left_on='Ticker', right_on='Ticker', how='left')
        chart_df['Market_Value'] = chart_df.apply(
            lambda row: row['Shares'] * row.get('Current_Price', 0) * 
            currency_rates.get(f"{row['Currency']}/{base_currency}", 1.0),
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
                    lambda row: row['Shares'] * row.get('Annual_Dividend', 0) * 
                    currency_rates.get(f"{row['Currency']}/{base_currency}", 1.0) if pd.notna(row.get('Annual_Dividend')) else 0,
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
                    row['Currency'],
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
            # Calendar year view (simplified - would need actual dividend schedule)
            st.markdown("**Kalenderår: Utdelningar per månad**")
            st.info("This visualization requires dividend schedule data. Currently showing estimated monthly distribution.")
            
            # Estimate monthly dividends (divide annual by 12, assuming quarterly payments)
            monthly_data = []
            for idx, row in chart_df.iterrows():
                annual_div = row.get('Dividend_Income', 0)
                if annual_div > 0:
                    # Assume quarterly payments (simplified)
                    quarterly = annual_div / 4
                    for month in [3, 6, 9, 12]:  # Typical dividend months
                        monthly_data.append({
                            'Month': month,
                            'Ticker': row['Ticker'],
                            'Dividend': quarterly
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
                    title='Estimated Monthly Dividend Income',
                    labels={'Dividend': f'Dividend ({base_currency})', 'Month_Name': 'Month'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
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
    """Display transaction history"""
    
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
                except Exception as e:
                    st.error(f"❌ Error adding transaction: {str(e)}")


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


if __name__ == "__main__":
    main()


