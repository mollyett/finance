# Stock Dividend Portfolio Tracker

A comprehensive Streamlit web application for tracking your stock portfolio with dividend analysis and multi-currency support.

## Features

- 📊 **Portfolio Tracking**: Track your stock holdings with real-time price updates
- 💵 **Dividend Analysis**: Calculate annual, quarterly, and monthly dividend income
- 💱 **Multi-Currency Support**: Automatic currency conversion (SEK, USD, EUR)
- 📈 **Visualizations**: Interactive charts showing portfolio allocation and dividend yields
- 📤 **CSV Import/Export**: Upload your portfolio data or download a template
- 🔒 **Data Privacy**: Option to upload your own CSV without storing data on the server
- 📱 **Responsive Design**: Works on both desktop and mobile devices

## Tech Stack

- **Frontend/Backend**: Streamlit
- **Data Source**: yfinance (Yahoo Finance API)
- **Visualization**: Plotly
- **Storage**: CSV-based (no database required)

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

### Option 1: Upload CSV File

1. Click "Download Template" in the sidebar to get a sample CSV file
2. Fill in your portfolio data:
   - **Ticker**: Stock symbol (e.g., AAPL, MSFT)
   - **Shares**: Number of shares owned
   - **Avg_Price**: Average purchase price per share
   - **Currency**: Currency of the stock (SEK, USD, or EUR)
3. Upload your CSV file using the file uploader

### Option 2: Manual Entry

1. Select "Enter Manually" in the sidebar
2. Enter the number of stocks you want to add
3. Fill in the details for each stock
4. Click "Add to Portfolio"

## CSV Format

Your CSV file should have the following columns:

| Ticker | Shares | Avg_Price | Currency |
|--------|--------|-----------|----------|
| AAPL   | 10     | 150.00    | USD      |
| MSFT   | 5      | 300.00    | USD      |
| VOLV-B | 20     | 200.00    | SEK      |

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

## Notes

- Data is cached for 5 minutes (stocks) and 1 hour (currency rates) to improve performance
- The app uses yfinance which may have rate limits for free usage
- Some stocks may not have dividend information available

## License

MIT License

