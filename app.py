"""
Stock Dividend Portfolio Tracker - Main Application
A Streamlit app for tracking stock dividends with currency conversion support.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import io

from data_fetcher import fetch_stock_data, fetch_currency_rates
from portfolio_calculator import (
    calculate_portfolio_metrics, 
    calculate_dividend_yield,
    calculate_snowball_effect,
    calculate_advanced_statistics
)
from utils import create_template_csv, validate_portfolio_data

# Page configuration
st.set_page_config(
    page_title="Stock Dividend Portfolio Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive design
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
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    st.title("📈 Stock Dividend Portfolio Tracker")
    st.markdown("---")
    
    # Sidebar for user options
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Currency selection
        base_currency = st.selectbox(
            "Base Currency",
            ["SEK", "USD", "EUR"],
            index=0,
            help="Select your base currency for calculations"
        )
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Upload CSV", "Enter Manually", "Manage Holdings"],
            help="Choose how to input your portfolio data"
        )
        
        # Clear portfolio button
        if st.session_state.portfolio_df is not None:
            if st.button("🗑️ Clear Portfolio", use_container_width=True):
                st.session_state.portfolio_df = None
                st.session_state.editing_mode = False
                st.rerun()
        
        # Template CSV download
        st.markdown("---")
        st.subheader("📥 Template CSV")
        template_csv = create_template_csv()
        st.download_button(
            label="Download Template",
            data=template_csv,
            file_name="portfolio_template.csv",
            mime="text/csv",
            help="Download a template CSV file to format your portfolio data"
        )
    
    # Initialize session state
    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = None
    if 'currency_rates' not in st.session_state:
        st.session_state.currency_rates = {}
    if 'editing_mode' not in st.session_state:
        st.session_state.editing_mode = False
    
    # Data input section
    if data_source == "Upload CSV":
        portfolio_df = handle_csv_upload()
    elif data_source == "Enter Manually":
        portfolio_df = handle_manual_input()
    else:  # Manage Holdings
        portfolio_df = handle_manage_holdings()
    
    if portfolio_df is not None and not portfolio_df.empty:
        # Validate data
        validation_result = validate_portfolio_data(portfolio_df)
        if not validation_result['valid']:
            st.error(f"❌ Data validation error: {validation_result['error']}")
            return
        
        st.session_state.portfolio_df = portfolio_df
        
        # Fetch currency rates
        with st.spinner("Fetching currency exchange rates..."):
            currency_rates = fetch_currency_rates()
            st.session_state.currency_rates = currency_rates
        
        # Display portfolio dashboard
        display_portfolio_dashboard(portfolio_df, base_currency, currency_rates)
    else:
        st.info("👆 Please upload a CSV file or enter your portfolio data manually to get started.")


def handle_csv_upload():
    """Handle CSV file upload"""
    st.subheader("📤 Upload Portfolio CSV")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your portfolio CSV file. Use the template for correct formatting."
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} stocks from CSV")
            return df
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            return None
    return None


def handle_manual_input():
    """Handle manual portfolio data input"""
    st.subheader("✍️ Enter Portfolio Data")
    
    with st.form("portfolio_form"):
        num_stocks = st.number_input(
            "Number of stocks",
            min_value=1,
            max_value=100,
            value=1,
            step=1
        )
        
        stocks_data = []
        cols = st.columns(4)
        
        for i in range(num_stocks):
            with st.expander(f"Stock {i+1}", expanded=(i == 0)):
                ticker = st.text_input(f"Ticker Symbol {i+1}", key=f"ticker_{i}", placeholder="AAPL")
                shares = st.number_input(f"Shares {i+1}", min_value=0.0, value=0.0, step=0.01, key=f"shares_{i}")
                avg_price = st.number_input(f"Avg Price {i+1}", min_value=0.0, value=0.0, step=0.01, key=f"price_{i}")
                currency = st.selectbox(f"Currency {i+1}", ["SEK", "USD", "EUR"], key=f"currency_{i}")
                
                if ticker and shares > 0 and avg_price > 0:
                    stocks_data.append({
                        'Ticker': ticker.upper(),
                        'Shares': shares,
                        'Avg_Price': avg_price,
                        'Currency': currency
                    })
        
        submitted = st.form_submit_button("Add to Portfolio")
        
        if submitted and stocks_data:
            df = pd.DataFrame(stocks_data)
            # Merge with existing portfolio if available
            if st.session_state.portfolio_df is not None:
                existing_df = st.session_state.portfolio_df
                # Combine dataframes, handling duplicates by updating shares
                combined_df = existing_df.copy()
                for new_row in stocks_data:
                    existing_idx = combined_df[combined_df['Ticker'] == new_row['Ticker']].index
                    if len(existing_idx) > 0:
                        # Update existing position
                        combined_df.loc[existing_idx[0], 'Shares'] += new_row['Shares']
                        # Recalculate average price (weighted average)
                        old_shares = combined_df.loc[existing_idx[0], 'Shares'] - new_row['Shares']
                        old_price = combined_df.loc[existing_idx[0], 'Avg_Price']
                        new_shares = new_row['Shares']
                        new_price = new_row['Avg_Price']
                        total_shares = old_shares + new_shares
                        if total_shares > 0:
                            combined_df.loc[existing_idx[0], 'Avg_Price'] = (
                                (old_shares * old_price + new_shares * new_price) / total_shares
                            )
                    else:
                        # Add new position
                        combined_df = pd.concat([combined_df, pd.DataFrame([new_row])], ignore_index=True)
                st.success(f"✅ Added/Updated {len(stocks_data)} stocks. Total positions: {len(combined_df)}")
                return combined_df
            else:
                st.success(f"✅ Added {len(df)} stocks to portfolio")
                return df
    
    return st.session_state.portfolio_df if st.session_state.portfolio_df is not None else None


def handle_manage_holdings():
    """Handle portfolio holdings management (add/edit/remove)"""
    st.subheader("📝 Manage Portfolio Holdings")
    
    if st.session_state.portfolio_df is None or st.session_state.portfolio_df.empty:
        st.info("No portfolio data available. Please upload a CSV or enter data manually first.")
        return None
    
    # Display current holdings
    st.write("**Current Holdings:**")
    current_df = st.session_state.portfolio_df.copy()
    st.dataframe(current_df, use_container_width=True, hide_index=True)
    
    # Add new holding
    with st.expander("➕ Add New Holding", expanded=False):
        with st.form("add_holding_form"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                new_ticker = st.text_input("Ticker", key="new_ticker", placeholder="AAPL")
            with col2:
                new_shares = st.number_input("Shares", min_value=0.0, value=0.0, step=0.01, key="new_shares")
            with col3:
                new_price = st.number_input("Avg Price", min_value=0.0, value=0.0, step=0.01, key="new_price")
            with col4:
                new_currency = st.selectbox("Currency", ["SEK", "USD", "EUR"], key="new_currency")
            
            if st.form_submit_button("Add Holding"):
                if new_ticker and new_shares > 0 and new_price > 0:
                    new_row = pd.DataFrame([{
                        'Ticker': new_ticker.upper(),
                        'Shares': new_shares,
                        'Avg_Price': new_price,
                        'Currency': new_currency
                    }])
                    current_df = pd.concat([current_df, new_row], ignore_index=True)
                    st.success(f"✅ Added {new_ticker.upper()}")
                    st.session_state.portfolio_df = current_df
                    st.rerun()
    
    # Edit/Remove holdings
    with st.expander("✏️ Edit or Remove Holdings", expanded=False):
        if len(current_df) > 0:
            selected_ticker = st.selectbox(
                "Select holding to edit/remove",
                current_df['Ticker'].tolist(),
                key="edit_ticker"
            )
            
            if selected_ticker:
                selected_row = current_df[current_df['Ticker'] == selected_ticker].iloc[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Current values for {selected_ticker}:**")
                    st.write(f"- Shares: {selected_row['Shares']}")
                    st.write(f"- Avg Price: {selected_row['Avg_Price']}")
                    st.write(f"- Currency: {selected_row['Currency']}")
                
                with col2:
                    with st.form("edit_holding_form"):
                        edit_shares = st.number_input(
                            "New Shares",
                            min_value=0.0,
                            value=float(selected_row['Shares']),
                            step=0.01,
                            key="edit_shares"
                        )
                        edit_price = st.number_input(
                            "New Avg Price",
                            min_value=0.0,
                            value=float(selected_row['Avg_Price']),
                            step=0.01,
                            key="edit_price"
                        )
                        edit_currency = st.selectbox(
                            "Currency",
                            ["SEK", "USD", "EUR"],
                            index=["SEK", "USD", "EUR"].index(selected_row['Currency']),
                            key="edit_currency"
                        )
                        
                        col_edit, col_remove = st.columns(2)
                        with col_edit:
                            if st.form_submit_button("💾 Update", use_container_width=True):
                                current_df.loc[current_df['Ticker'] == selected_ticker, 'Shares'] = edit_shares
                                current_df.loc[current_df['Ticker'] == selected_ticker, 'Avg_Price'] = edit_price
                                current_df.loc[current_df['Ticker'] == selected_ticker, 'Currency'] = edit_currency
                                st.session_state.portfolio_df = current_df
                                st.success(f"✅ Updated {selected_ticker}")
                                st.rerun()
                        
                        with col_remove:
                            if st.form_submit_button("🗑️ Remove", use_container_width=True):
                                current_df = current_df[current_df['Ticker'] != selected_ticker]
                                st.session_state.portfolio_df = current_df
                                st.success(f"✅ Removed {selected_ticker}")
                                st.rerun()
    
    return current_df


def display_portfolio_dashboard(df, base_currency, currency_rates):
    """Display the main portfolio dashboard"""
    
    st.markdown("---")
    st.header("📊 Portfolio Dashboard")
    
    # Fetch stock data
    with st.spinner("Fetching stock data... This may take a moment."):
        portfolio_data = fetch_stock_data(df)
    
    if portfolio_data is None or portfolio_data.empty:
        st.error("❌ Failed to fetch stock data. Please check your ticker symbols.")
        return
    
    # Calculate portfolio metrics
    metrics = calculate_portfolio_metrics(
        portfolio_data,
        df,
        base_currency,
        currency_rates
    )
    
    # Display key metrics
    display_key_metrics(metrics, base_currency)
    
    # Portfolio table
    display_portfolio_table(portfolio_data, df, base_currency, currency_rates)
    
    # Advanced Statistics
    display_advanced_statistics(portfolio_data, df, base_currency, currency_rates)
    
    # Dividend analysis
    display_dividend_analysis(portfolio_data, df, base_currency, currency_rates)
    
    # Snowball Effect
    display_snowball_effect(portfolio_data, df, base_currency, currency_rates)
    
    # Charts
    display_charts(portfolio_data, df, base_currency, currency_rates)


def display_key_metrics(metrics, base_currency):
    """Display key portfolio metrics"""
    
    st.subheader("💰 Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Value",
            f"{metrics['total_value']:,.2f} {base_currency}",
            delta=f"{metrics['total_value_change']:,.2f} {base_currency}" if metrics.get('total_value_change') else None
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


def display_portfolio_table(portfolio_data, df, base_currency, currency_rates):
    """Display detailed portfolio table"""
    
    st.subheader("📋 Portfolio Holdings")
    
    # Merge portfolio data with user data
    display_df = df.merge(portfolio_data, left_on='Ticker', right_on='Ticker', how='left')
    
    # Calculate values in base currency
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
    
    # Calculate dividend yield
    display_df['Annual_Dividend_Base'] = display_df.apply(
        lambda row: row.get('Annual_Dividend', 0) * currency_rates.get(f"{row['Currency']}/{base_currency}", 1.0) if pd.notna(row.get('Annual_Dividend')) else 0,
        axis=1
    )
    display_df['Dividend_Yield'] = (display_df['Annual_Dividend_Base'] / display_df['Current_Price_Base'] * 100).round(2)
    display_df['Dividend_Income'] = display_df['Shares'] * display_df['Annual_Dividend_Base']
    
    # Select columns to display
    display_columns = [
        'Ticker', 'Shares', 'Current_Price_Base', 'Avg_Price_Base',
        'Market_Value', 'Cost_Basis', 'Gain_Loss', 'Gain_Loss_Pct',
        'Dividend_Yield', 'Dividend_Income'
    ]
    
    # Format the dataframe
    formatted_df = display_df[display_columns].copy()
    formatted_df.columns = [
        'Ticker', 'Shares', f'Current Price ({base_currency})', f'Avg Price ({base_currency})',
        'Market Value', 'Cost Basis', 'Gain/Loss', 'Gain/Loss %',
        'Dividend Yield %', 'Annual Dividend Income'
    ]
    
    # Format numeric columns
    for col in formatted_df.columns[1:]:
        if col != 'Ticker':
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
    
    st.dataframe(formatted_df, use_container_width=True, hide_index=True)


def display_dividend_analysis(portfolio_data, df, base_currency, currency_rates):
    """Display dividend analysis section"""
    
    st.subheader("💵 Dividend Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate total dividend income
        total_dividend_income = 0
        for idx, row in df.iterrows():
            ticker_data = portfolio_data[portfolio_data['Ticker'] == row['Ticker']]
            if not ticker_data.empty:
                annual_div = ticker_data.iloc[0].get('Annual_Dividend', 0)
                if pd.notna(annual_div):
                    rate = currency_rates.get(f"{row['Currency']}/{base_currency}", 1.0)
                    total_dividend_income += row['Shares'] * annual_div * rate
        
        st.metric(
            "Total Annual Dividend Income",
            f"{total_dividend_income:,.2f} {base_currency}"
        )
        
        # Portfolio dividend yield
        total_value = sum(
            row['Shares'] * portfolio_data[portfolio_data['Ticker'] == row['Ticker']].iloc[0].get('Current_Price', 0) * 
            currency_rates.get(f"{row['Currency']}/{base_currency}", 1.0)
            for idx, row in df.iterrows()
            if not portfolio_data[portfolio_data['Ticker'] == row['Ticker']].empty
        )
        
        portfolio_yield = (total_dividend_income / total_value * 100) if total_value > 0 else 0
        st.metric(
            "Portfolio Dividend Yield",
            f"{portfolio_yield:.2f}%"
        )
    
    with col2:
        # Monthly dividend income
        monthly_income = total_dividend_income / 12
        st.metric(
            "Estimated Monthly Dividend Income",
            f"{monthly_income:,.2f} {base_currency}"
        )
        
        # Quarterly dividend income
        quarterly_income = total_dividend_income / 4
        st.metric(
            "Estimated Quarterly Dividend Income",
            f"{quarterly_income:,.2f} {base_currency}"
        )


def display_advanced_statistics(portfolio_data, df, base_currency, currency_rates):
    """Display advanced portfolio statistics"""
    
    st.subheader("📊 Avancerad Statistik")
    
    stats = calculate_advanced_statistics(portfolio_data, df, base_currency, currency_rates)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Antal Positioner", stats['num_positions'])
        st.metric("Totalt Antal Aktier", f"{stats['total_shares']:,.0f}")
        if stats['largest_position']:
            st.metric(
                "Största Position",
                f"{stats['largest_position']['ticker']}",
                delta=f"{stats['largest_position']['percentage']:.1f}%"
            )
    
    with col2:
        st.metric(
            "Viktad Genomsnittlig Dividend Yield",
            f"{stats['weighted_avg_dividend_yield']:.2f}%"
        )
        if stats['smallest_position']:
            st.metric(
                "Minsta Position",
                f"{stats['smallest_position']['ticker']}",
                delta=f"{stats['smallest_position']['percentage']:.1f}%"
            )
        if stats['portfolio_concentration']:
            st.metric(
                "Top 5 Koncentration",
                f"{stats['portfolio_concentration']['top_5_percentage']:.1f}%"
            )
    
    with col3:
        if stats['top_dividend_payers']:
            st.write("**Top 5 Dividendbetalare:**")
            for i, payer in enumerate(stats['top_dividend_payers'], 1):
                st.write(
                    f"{i}. **{payer['ticker']}**: "
                    f"{payer['dividend_income']:,.2f} {base_currency} "
                    f"(Yield: {payer['dividend_yield']:.2f}%)"
                )


def display_snowball_effect(portfolio_data, df, base_currency, currency_rates):
    """Display snowball effect visualization"""
    
    st.subheader("❄️ Snöbollseffekt (Dividend Compounding)")
    st.markdown("*Visa hur din portfölj växer när du återinvesterar utdelningar*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        years = st.slider("Antal år att projicera", 1, 30, 10, 1)
    with col2:
        reinvest = st.checkbox("Återinvestera utdelningar", value=True)
    with col3:
        growth_rate = st.slider("Årlig tillväxt (%)", 0.0, 15.0, 5.0, 0.5) / 100
    
    if st.button("🔄 Beräkna Snöbollseffekt", use_container_width=True):
        with st.spinner("Beräknar snöbollseffekt..."):
            projections = calculate_snowball_effect(
                portfolio_data,
                df,
                base_currency,
                currency_rates,
                years=years,
                reinvest_dividends=reinvest,
                annual_growth_rate=growth_rate
            )
            
            if not projections.empty:
                # Display key metrics
                initial_value = projections.iloc[0]['Portfolio_Value']
                final_value = projections.iloc[-1]['Portfolio_Value']
                total_dividends = projections.iloc[-1]['Total_Dividends_Received']
                final_annual_dividend = projections.iloc[-1]['Annual_Dividend_Income']
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric(
                        "Initialt Värde",
                        f"{initial_value:,.2f} {base_currency}"
                    )
                
                with metric_col2:
                    st.metric(
                        f"Värde efter {years} år",
                        f"{final_value:,.2f} {base_currency}",
                        delta=f"{((final_value - initial_value) / initial_value * 100):.1f}%"
                    )
                
                with metric_col3:
                    st.metric(
                        "Totala Utdelningar",
                        f"{total_dividends:,.2f} {base_currency}"
                    )
                
                with metric_col4:
                    st.metric(
                        f"Årlig Utdelning år {years}",
                        f"{final_annual_dividend:,.2f} {base_currency}",
                        delta=f"{((final_annual_dividend / projections.iloc[0]['Annual_Dividend_Income'] - 1) * 100):.1f}%" if projections.iloc[0]['Annual_Dividend_Income'] > 0 else None
                    )
                
                # Visualization
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Portföljvärde över tid', 'Årlig utdelningsinkomst över tid'),
                        vertical_spacing=0.15
                    )
                    
                    # Portfolio value over time
                    fig.add_trace(
                        go.Scatter(
                            x=projections['Year'],
                            y=projections['Portfolio_Value'],
                            mode='lines+markers',
                            name='Portföljvärde',
                            line=dict(color='#1f77b4', width=3),
                            fill='tonexty'
                        ),
                        row=1, col=1
                    )
                    
                    # Annual dividend income over time
                    fig.add_trace(
                        go.Scatter(
                            x=projections['Year'],
                            y=projections['Annual_Dividend_Income'],
                            mode='lines+markers',
                            name='Årlig utdelning',
                            line=dict(color='#2ca02c', width=3),
                            fill='tonexty'
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_xaxes(title_text="År", row=2, col=1)
                    fig.update_yaxes(title_text=f"Värde ({base_currency})", row=1, col=1)
                    fig.update_yaxes(title_text=f"Utdelning ({base_currency})", row=2, col=1)
                    
                    fig.update_layout(
                        height=700,
                        title_text="Snöbollseffekt: Portföljutveckling med återinvesterade utdelningar",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    with st.expander("📋 Detaljerad Projicering", expanded=False):
                        display_proj = projections.copy()
                        display_proj.columns = [
                            'År', 'Portföljvärde', 'Årlig Utdelning',
                            'Totala Utdelningar', 'Kumulativ Avkastning (%)'
                        ]
                        for col in display_proj.columns[1:]:
                            display_proj[col] = display_proj[col].apply(lambda x: f"{x:,.2f}")
                        st.dataframe(display_proj, use_container_width=True, hide_index=True)
                    
                except ImportError:
                    st.warning("Plotly krävs för visualisering. Installera med: pip install plotly")
                    st.dataframe(projections, use_container_width=True)


def display_charts(portfolio_data, df, base_currency, currency_rates):
    """Display portfolio charts"""
    
    st.subheader("📈 Visualiseringar")
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Prepare data for charts
        chart_df = df.merge(portfolio_data, left_on='Ticker', right_on='Ticker', how='left')
        chart_df['Market_Value'] = chart_df.apply(
            lambda row: row['Shares'] * row.get('Current_Price', 0) * 
            currency_rates.get(f"{row['Currency']}/{base_currency}", 1.0),
            axis=1
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Portfolio allocation pie chart
            fig_pie = px.pie(
                chart_df,
                values='Market_Value',
                names='Ticker',
                title='Portföljfördelning efter Värde'
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
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
                title='Dividend Yield per Aktie',
                labels={'Dividend_Yield': 'Dividend Yield (%)'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
    except ImportError:
        st.warning("Plotly är inte installerat. Diagram är inaktiverade. Installera med: pip install plotly")


if __name__ == "__main__":
    main()

