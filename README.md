# Stock Dividend Portfolio Tracker

A comprehensive Streamlit web application for tracking your stock portfolio with dividend analysis and multi-currency support.

## Features

### Core Functionality
- 📊 **Portfolio Tracking**: Track your stock holdings with real-time price updates via yfinance
- 💾 **SQLite Database**: Persistent local storage of all transactions and holdings
- 💵 **Dividend Analysis**: Calculate annual, quarterly, and monthly dividend income
- 💱 **Multi-Currency Support**: Automatic currency conversion (SEK, USD, EUR) using real-time exchange rates
- 📈 **Advanced Visualizations**: Interactive Plotly charts for comprehensive portfolio analysis

### Transaction Management
- ➕ **Add Transactions**: Record purchases with date, price, quantity, sector, country, and target allocation
- 📜 **Transaction History**: View complete history of all transactions
- 🎯 **Target Allocations**: Set target allocation percentages for rebalancing

### Advanced Metrics
- **Yield on Cost (YoY)**: Shows dividend yield based on purchase price
- **Portfolio Yield**: Current dividend yield of entire portfolio
- **Payout Ratio**: Company's dividend payout ratio (where available)
- **Gain/Loss Tracking**: Real-time profit/loss calculations with percentages

### Rebalancing
- ⚖️ **Rebalance Suggestions**: Automated suggestions based on target vs. current allocations
- **Buy/Sell Recommendations**: Calculated share amounts needed to reach target allocations

### Visualizations
- **Portfolio Allocation**: Pie charts for value and dividend income distribution
- **Dividend Yield Charts**: Bar charts showing yield by stock
- **Snowball Effect**: Projected portfolio growth with dividend reinvestment over 10-30 years
- **Calendar Year View**: Monthly dividend income heatmap/bar chart
- **Sector Distribution**: Pie chart showing sector allocation
- **Geographic Exposure**: Pie chart showing country/region allocation

## Tech Stack

- **Frontend/Backend**: Streamlit
- **Data Source**: yfinance (Yahoo Finance API)
- **Visualization**: Plotly
- **Storage**: SQLite database (local, persistent)
- **Calculations**: Pandas & NumPy

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd finance
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Deployment to Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your repository
5. Set the main file path to `app.py`
6. Click "Deploy"

The app will automatically use the `requirements.txt` and `.streamlit/config.toml` files.

## Usage

### Getting Started

1. **Add Your First Transaction**:
   - Navigate to "➕ Add Transaction" in the sidebar
   - Fill in the required fields:
     - **Ticker**: Stock symbol (e.g., AAPL, investor-b.st)
     - **Purchase Date**: Date of purchase
     - **Purchase Price**: Price per share
     - **Quantity**: Number of shares
     - **Currency**: SEK, USD, or EUR
   - Optional fields:
     - **Sector**: Industry sector (e.g., Technology, Finance)
     - **Country**: Country of listing (e.g., USA, Sweden)
     - **Target Allocation %**: Desired portfolio weight for rebalancing
   - Click "Add Transaction"

2. **View Your Portfolio**:
   - Go to "📊 Overview" to see your dashboard
   - View key metrics, holdings table, and visualizations

3. **Set Target Allocations**:
   - Navigate to "⚙️ Settings"
   - Set target allocation percentages for each holding
   - Return to Overview to see rebalancing suggestions

### Navigation

- **📊 Overview**: Main dashboard with metrics, holdings, charts, and transaction history
- **➕ Add Transaction**: Add new stock purchases to your portfolio
- **⚙️ Settings**: Configure target allocations and export data

## Ticker Format

The app supports tickers from various exchanges:
- **US Stocks**: AAPL, MSFT, GOOGL
- **Swedish Stocks**: investor-b.st, volv-b.st (use .ST suffix)
- **Other Exchanges**: Supports .OL (Oslo), .CO (Copenhagen), etc.

Note: For Stockholm exchange stocks, use the `.ST` suffix (e.g., `investor-b.st`).

## Features Explained

### Currency Conversion
The app automatically fetches real-time exchange rates for:
- SEK/USD and USD/SEK
- SEK/EUR and EUR/SEK
- USD/EUR and EUR/USD

All portfolio values are converted to your selected base currency for easy comparison.

### Dividend Analysis
- **Annual Dividend Income**: Total dividends you'll receive per year
- **Portfolio Dividend Yield**: Average dividend yield across your portfolio
- **Monthly/Quarterly Estimates**: Breakdown of dividend income by period

### Performance Metrics
- **Total Value**: Current market value of your portfolio
- **Total Cost**: Your total investment (cost basis)
- **Gain/Loss**: Profit or loss in both absolute and percentage terms

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## Database

The app uses a local SQLite database (`.data/portfolio.db`) to store:
- All transactions with full history
- Holdings summary (automatically calculated from transactions)
- Target allocations

The database is created automatically on first run. Data persists between sessions.

## Notes

- Data is cached for 5 minutes (stocks) and 1 hour (currency rates) to improve performance
- The app uses yfinance which may have rate limits for free usage
- Some stocks may not have dividend information available
- Dividend schedule data (for calendar year view) is estimated based on annual dividends
- Historical dividend data for CAGR calculations would require additional data sources

## License

MIT License

