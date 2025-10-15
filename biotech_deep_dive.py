"""
Advanced Biotech Analysis - Deep Dive
Comparing CRISPR stocks, risk metrics, correlations, backtesting, and news
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print(" " * 30 + "BIOTECH DEEP DIVE ANALYSIS")
print("=" * 100)

# =============================================================================
# ANALYSIS 1: CRISPR GENE EDITING SHOWDOWN (NTLA vs EDIT vs CRSP)
# =============================================================================
print("\n\n" + "=" * 100)
print("ANALYSIS 1: CRISPR GENE EDITING STOCKS COMPARISON")
print("=" * 100)
print("\nComparing the three major CRISPR gene editing companies...\n")

crispr_tickers = ['NTLA', 'EDIT', 'CRSP']

# Download 1 year of data for detailed comparison
print("Downloading 1 year of data for NTLA, EDIT, CRSP...")
crispr_data = yf.download(crispr_tickers, period='1y', interval='1d', progress=False)

# Get company info
comparison = []
for ticker in crispr_tickers:
    stock = yf.Ticker(ticker)
    info = stock.info
    
    comparison.append({
        'Ticker': ticker,
        'Company': info.get('longName', 'N/A'),
        'Price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
        'Market Cap': info.get('marketCap', 'N/A'),
        'Employees': info.get('fullTimeEmployees', 'N/A'),
        'City': info.get('city', 'N/A'),
        'Description': info.get('longBusinessSummary', 'N/A')[:200] + '...'
    })

df_comparison = pd.DataFrame(comparison)

print("\n" + "-" * 100)
print("Company Profiles:")
print("-" * 100)
for _, row in df_comparison.iterrows():
    print(f"\n{row['Ticker']} - {row['Company']}")
    print(f"  Price: ${row['Price']:.2f}")
    print(f"  Market Cap: ${row['Market Cap']/1e9:.2f}B" if row['Market Cap'] != 'N/A' else "  Market Cap: N/A")
    print(f"  Employees: {row['Employees']:,}" if row['Employees'] != 'N/A' else "  Employees: N/A")
    print(f"  Location: {row['City']}")
    print(f"  Business: {row['Description']}")

# Performance comparison
print("\n\n" + "-" * 100)
print("Performance Head-to-Head (Last 12 Months):")
print("-" * 100)

perf_metrics = []
for ticker in crispr_tickers:
    closes = crispr_data['Close'][ticker].dropna()
    
    if len(closes) > 0:
        current = closes.iloc[-1]
        start = closes.iloc[0]
        
        # Different time periods
        return_1y = ((current - start) / start) * 100
        return_6m = ((current - closes.iloc[-120 if len(closes) >= 120 else 0]) / closes.iloc[-120 if len(closes) >= 120 else 0]) * 100
        return_3m = ((current - closes.iloc[-60 if len(closes) >= 60 else 0]) / closes.iloc[-60 if len(closes) >= 60 else 0]) * 100
        return_1m = ((current - closes.iloc[-20 if len(closes) >= 20 else 0]) / closes.iloc[-20 if len(closes) >= 20 else 0]) * 100
        
        # Volatility
        daily_returns = closes.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Max drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Best and worst day
        best_day = daily_returns.max() * 100
        worst_day = daily_returns.min() * 100
        
        perf_metrics.append({
            'Ticker': ticker,
            '1Y Return %': return_1y,
            '6M Return %': return_6m,
            '3M Return %': return_3m,
            '1M Return %': return_1m,
            'Volatility %': volatility,
            'Max Drawdown %': max_drawdown,
            'Best Day %': best_day,
            'Worst Day %': worst_day
        })

df_perf = pd.DataFrame(perf_metrics)
print("\n", df_perf.to_string(index=False))

print("\n" + "-" * 100)
print("Winner Analysis:")
print("-" * 100)
for period in ['1Y Return %', '6M Return %', '3M Return %', '1M Return %']:
    winner = df_perf.loc[df_perf[period].idxmax()]
    print(f"{period:15s}: {winner['Ticker']} ({winner[period]:+.2f}%)")

# =============================================================================
# ANALYSIS 2: RISK-ADJUSTED RETURNS (SHARPE RATIOS)
# =============================================================================
print("\n\n" + "=" * 100)
print("ANALYSIS 2: RISK-ADJUSTED RETURNS (SHARPE RATIO)")
print("=" * 100)
print("\nSharpe Ratio = (Return - Risk-Free Rate) / Volatility")
print("Higher is better. Above 1.0 is good, above 2.0 is excellent.\n")

# Top performers from our original analysis
top_tickers = ['NTLA', 'SANA', 'EDIT', 'ARWR', 'IONS', 'INSM', 'ALNY', 'GILD', 'VRTX', 'REGN']

print("Downloading 1 year of data for top performers...")
sharpe_data = yf.download(top_tickers, period='1y', interval='1d', progress=False)

risk_free_rate = 0.045  # Assuming 4.5% risk-free rate (current T-bill rate)

sharpe_results = []
for ticker in top_tickers:
    try:
        closes = sharpe_data['Close'][ticker].dropna() if len(top_tickers) > 1 else sharpe_data['Close'].dropna()
        
        if len(closes) > 20:
            daily_returns = closes.pct_change().dropna()
            
            # Calculate metrics
            annual_return = ((closes.iloc[-1] / closes.iloc[0]) - 1) * 100
            volatility = daily_returns.std() * np.sqrt(252)
            avg_daily_return = daily_returns.mean()
            annual_avg_return = avg_daily_return * 252
            
            # Sharpe ratio
            sharpe = (annual_avg_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino ratio (only penalizes downside volatility)
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino = (annual_avg_return - risk_free_rate) / downside_std if downside_std > 0 else 0
            
            # Calmar ratio (return / max drawdown)
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            calmar = annual_avg_return / max_drawdown if max_drawdown > 0 else 0
            
            sharpe_results.append({
                'Ticker': ticker,
                'Annual Return %': annual_return,
                'Volatility %': volatility * 100,
                'Sharpe Ratio': sharpe,
                'Sortino Ratio': sortino,
                'Calmar Ratio': calmar,
                'Max Drawdown %': max_drawdown * 100
            })
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)[:50]}")
        continue

df_sharpe = pd.DataFrame(sharpe_results)
df_sharpe = df_sharpe.sort_values('Sharpe Ratio', ascending=False)

print("\n" + "-" * 100)
print("Risk-Adjusted Performance Rankings:")
print("-" * 100)
print(df_sharpe.to_string(index=False))

print("\n" + "-" * 100)
print("Interpretation:")
print("-" * 100)
print(f"Best Sharpe Ratio:  {df_sharpe.iloc[0]['Ticker']} ({df_sharpe.iloc[0]['Sharpe Ratio']:.2f}) - Best risk-adjusted returns")
print(f"Best Sortino Ratio: {df_sharpe.loc[df_sharpe['Sortino Ratio'].idxmax()]['Ticker']} - Best downside-protected returns")
print(f"Lowest Drawdown:    {df_sharpe.loc[df_sharpe['Max Drawdown %'].idxmin()]['Ticker']} - Most stable")

# =============================================================================
# ANALYSIS 3: CORRELATION MATRIX
# =============================================================================
print("\n\n" + "=" * 100)
print("ANALYSIS 3: CORRELATION MATRIX (Which Stocks Move Together?)")
print("=" * 100)
print("\nCorrelation: 1.0 = perfect positive, -1.0 = perfect negative, 0.0 = no relationship\n")

# Use a broader set for correlations
corr_tickers = ['NTLA', 'EDIT', 'CRSP', 'BEAM', 'VRTX', 'REGN', 'GILD', 'MRNA', 'BNTX', 'IONS']

print("Calculating correlations based on 6 months of daily returns...")
corr_data = yf.download(corr_tickers, period='6mo', interval='1d', progress=False)

# Calculate daily returns
returns_dict = {}
for ticker in corr_tickers:
    closes = corr_data['Close'][ticker].dropna() if len(corr_tickers) > 1 else corr_data['Close'].dropna()
    if len(closes) > 0:
        returns_dict[ticker] = closes.pct_change().dropna()

df_returns = pd.DataFrame(returns_dict)
correlation_matrix = df_returns.corr()

print("\n" + "-" * 100)
print("Correlation Matrix:")
print("-" * 100)
print(correlation_matrix.round(2).to_string())

print("\n" + "-" * 100)
print("Key Findings:")
print("-" * 100)

# Find highest correlations (excluding diagonal)
high_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        ticker1 = correlation_matrix.columns[i]
        ticker2 = correlation_matrix.columns[j]
        corr_value = correlation_matrix.iloc[i, j]
        high_corr.append((ticker1, ticker2, corr_value))

high_corr.sort(key=lambda x: abs(x[2]), reverse=True)

print("\nMost Correlated Pairs (move together):")
for ticker1, ticker2, corr in high_corr[:5]:
    print(f"  {ticker1} <-> {ticker2}: {corr:.2f}")

print("\nLeast Correlated Pairs (diversification opportunities):")
for ticker1, ticker2, corr in high_corr[-5:]:
    print(f"  {ticker1} <-> {ticker2}: {corr:.2f}")

# =============================================================================
# ANALYSIS 4: MOMENTUM STRATEGY BACKTEST
# =============================================================================
print("\n\n" + "=" * 100)
print("ANALYSIS 4: MOMENTUM STRATEGY BACKTEST")
print("=" * 100)
print("\nStrategy: Buy stocks when price > 50-day MA, Sell when price < 50-day MA")
print("Testing on: NTLA (best recent performer)\n")

# Download 2 years for better backtest
print("Downloading 2 years of NTLA data for backtesting...")
ntla = yf.Ticker('NTLA')
backtest_data = ntla.history(period='2y', interval='1d')

if len(backtest_data) > 0:
    df_bt = backtest_data.copy()
    
    # Calculate 50-day moving average
    df_bt['MA50'] = df_bt['Close'].rolling(window=50).mean()
    df_bt['MA20'] = df_bt['Close'].rolling(window=20).mean()
    
    # Generate signals
    df_bt['Position'] = 0  # 0 = cash, 1 = invested
    df_bt.loc[df_bt['Close'] > df_bt['MA50'], 'Position'] = 1
    df_bt.loc[df_bt['Close'] < df_bt['MA50'], 'Position'] = 0
    
    # Calculate strategy returns
    df_bt['Daily_Return'] = df_bt['Close'].pct_change()
    df_bt['Strategy_Return'] = df_bt['Position'].shift(1) * df_bt['Daily_Return']  # Use previous day's signal
    df_bt['Buy_Hold_Return'] = df_bt['Daily_Return']
    
    # Cumulative returns
    df_bt['Strategy_Cumulative'] = (1 + df_bt['Strategy_Return'].fillna(0)).cumprod()
    df_bt['Buy_Hold_Cumulative'] = (1 + df_bt['Buy_Hold_Return'].fillna(0)).cumprod()
    
    # Performance metrics
    strategy_total_return = (df_bt['Strategy_Cumulative'].iloc[-1] - 1) * 100
    buyhold_total_return = (df_bt['Buy_Hold_Cumulative'].iloc[-1] - 1) * 100
    
    # Count trades
    df_bt['Signal_Change'] = df_bt['Position'].diff()
    num_buys = len(df_bt[df_bt['Signal_Change'] == 1])
    num_sells = len(df_bt[df_bt['Signal_Change'] == -1])
    
    # Win rate
    df_bt['Trade_Return'] = 0.0
    in_position = False
    entry_price = 0
    trade_returns = []
    
    for idx, row in df_bt.iterrows():
        if row['Signal_Change'] == 1:  # Buy signal
            in_position = True
            entry_price = row['Close']
        elif row['Signal_Change'] == -1 and in_position:  # Sell signal
            exit_price = row['Close']
            trade_return = ((exit_price - entry_price) / entry_price) * 100
            trade_returns.append(trade_return)
            in_position = False
    
    winning_trades = len([r for r in trade_returns if r > 0])
    total_trades = len(trade_returns)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    print("-" * 100)
    print("Backtest Results (NTLA - Last 2 Years):")
    print("-" * 100)
    print(f"\nMomentum Strategy:")
    print(f"  Total Return:        {strategy_total_return:+.2f}%")
    print(f"  Number of Trades:    {num_buys}")
    print(f"  Win Rate:            {win_rate:.1f}% ({winning_trades}/{total_trades} winning trades)")
    print(f"  Average Trade:       {np.mean(trade_returns):.2f}%" if trade_returns else "  Average Trade: N/A")
    print(f"  Best Trade:          {max(trade_returns):.2f}%" if trade_returns else "  Best Trade: N/A")
    print(f"  Worst Trade:         {min(trade_returns):.2f}%" if trade_returns else "  Worst Trade: N/A")
    
    print(f"\nBuy & Hold:")
    print(f"  Total Return:        {buyhold_total_return:+.2f}%")
    
    print(f"\nStrategy vs Buy & Hold:")
    outperformance = strategy_total_return - buyhold_total_return
    print(f"  Outperformance:      {outperformance:+.2f}%")
    
    if outperformance > 0:
        print(f"\n  >>> Momentum strategy BEAT buy-and-hold by {outperformance:.2f}%!")
    else:
        print(f"\n  >>> Buy-and-hold BEAT momentum strategy by {abs(outperformance):.2f}%")
    
    # Recent signals
    print("\n" + "-" * 100)
    print("Recent Trading Signals:")
    print("-" * 100)
    recent_signals = df_bt[df_bt['Signal_Change'] != 0].tail(10)
    for idx, row in recent_signals.iterrows():
        signal_type = "BUY" if row['Signal_Change'] == 1 else "SELL"
        print(f"  {idx.strftime('%Y-%m-%d')}: {signal_type:4s} at ${row['Close']:.2f}")
    
    current_position = "INVESTED" if df_bt['Position'].iloc[-1] == 1 else "CASH"
    print(f"\nCurrent Position: {current_position}")
    print(f"Current Price: ${df_bt['Close'].iloc[-1]:.2f}")
    print(f"50-day MA: ${df_bt['MA50'].iloc[-1]:.2f}")

# =============================================================================
# ANALYSIS 5: NEWS & CATALYSTS FOR TOP PERFORMERS
# =============================================================================
print("\n\n" + "=" * 100)
print("ANALYSIS 5: NEWS & CATALYSTS FOR TOP PERFORMERS")
print("=" * 100)
print("\nPulling recent news for NTLA, SANA, EDIT (top 3 performers)...\n")

top_performers = ['NTLA', 'SANA', 'EDIT']

for ticker in top_performers:
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        print("-" * 100)
        print(f"{ticker} - Recent News:")
        print("-" * 100)
        
        if news and len(news) > 0:
            for i, article in enumerate(news[:5], 1):  # Show top 5 news items
                title = article.get('title', 'No title')
                publisher = article.get('publisher', 'Unknown')
                pub_date = datetime.fromtimestamp(article.get('providerPublishTime', 0))
                link = article.get('link', 'No link')
                
                print(f"\n{i}. {title}")
                print(f"   Source: {publisher}")
                print(f"   Date: {pub_date.strftime('%Y-%m-%d %H:%M')}")
                print(f"   Link: {link}")
        else:
            print("  No recent news available")
        
        print()
        
    except Exception as e:
        print(f"Error fetching news for {ticker}: {str(e)[:100]}")

# =============================================================================
# SUMMARY & RECOMMENDATIONS
# =============================================================================
print("\n\n" + "=" * 100)
print("EXECUTIVE SUMMARY & KEY TAKEAWAYS")
print("=" * 100)

print("""
1. CRISPR GENE EDITING BATTLE:
   - All three companies (NTLA, EDIT, CRSP) show high volatility (65-105% annually)
   - NTLA has been the clear winner recently with explosive gains
   - Technology risk is high - these are speculative investments

2. RISK-ADJUSTED PERFORMANCE:
   - Look at Sharpe ratios above to see which stocks give you the best return per unit of risk
   - Higher Sortino ratios mean the stock protects you better on the downside
   - Low drawdown stocks are more stable (good for conservative investors)

3. CORRELATION INSIGHTS:
   - Gene editing stocks (NTLA, EDIT, CRSP, BEAM) tend to move together
   - For diversification, pair gene editing with large cap biotechs (GILD, REGN)
   - MRNA and BNTX (mRNA vaccines) have their own correlation pattern

4. MOMENTUM STRATEGY:
   - The backtest shows whether timing entries/exits beats buy-and-hold
   - Win rate tells you how often the strategy is right
   - Current signal tells you what to do TODAY

5. NEWS & CATALYSTS:
   - Recent news can explain sudden price moves
   - Clinical trial results, FDA approvals, and partnerships drive biotech stocks
   - Use news to understand WHY stocks moved, not just HOW MUCH

INVESTMENT IMPLICATIONS:
- High returns come with high volatility in biotech
- Diversification across different biotech sub-sectors reduces risk
- Momentum strategies can work but require discipline to follow signals
- News-driven volatility creates both risks and opportunities
""")

print("=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)

