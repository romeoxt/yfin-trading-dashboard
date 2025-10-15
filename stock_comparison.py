"""
Stock Comparison Tool - Compare multiple stocks side-by-side
Perfect for choosing the best entry among similar stocks
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from advanced_indicators import TechnicalIndicators

def analyze_stock_for_comparison(ticker):
    """Get comprehensive analysis for a stock"""
    try:
        stock = yf.Ticker(ticker)
        
        # Try different periods if one fails
        df = None
        for period in ['3mo', '6mo', '1y']:
            try:
                df = stock.history(period=period, interval='1d')
                if not df.empty and len(df) >= 50:
                    break
            except:
                continue
        
        if df is None or df.empty or len(df) < 50:
            return None
        
        info = stock.info
        
        # Calculate indicators
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = TechnicalIndicators.RSI(df['Close'])
        
        macd_data = TechnicalIndicators.MACD(df['Close'])
        df['MACD'] = macd_data['MACD']
        df['MACD_Signal'] = macd_data['Signal']
        
        df['ADX'] = TechnicalIndicators.ADX(df['High'], df['Low'], df['Close'])
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        
        latest = df.iloc[-1]
        current_price = latest['Close']
        
        # Trading signal
        conditions_met = 0
        if latest['Close'] > latest['MA20'] > latest['MA50']:
            conditions_met += 1
        if 40 < latest['RSI'] < 60:
            conditions_met += 1
        if latest['MACD'] > latest['MACD_Signal']:
            conditions_met += 1
        if latest['ADX'] > 25:
            conditions_met += 1
        if latest['Volume'] > latest['Volume_MA20']:
            conditions_met += 1
        
        score = conditions_met
        
        if score >= 4:
            tech_signal = "BUY"
        elif score >= 3:
            tech_signal = "NEUTRAL"
        else:
            tech_signal = "SELL"
        
        # Analyst rating
        analyst_rating = info.get('recommendationKey', 'N/A').upper() if info.get('recommendationKey') else 'N/A'
        
        # Alignment
        analyst_bullish = analyst_rating in ['STRONG_BUY', 'BUY']
        tech_bullish = tech_signal == "BUY"
        tech_bearish = tech_signal == "SELL"
        
        if analyst_bullish and tech_bullish:
            alignment = "ALIGNED-BUY"
        elif analyst_bullish and tech_bearish:
            alignment = "MIXED-WAIT"
        elif analyst_bullish:
            alignment = "NEUTRAL"
        else:
            alignment = "NEUTRAL"
        
        # Support/resistance
        recent_low = df['Low'].iloc[-20:].min()
        support = round(recent_low * 0.99, 2)
        
        # Performance
        perf_1m = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100 if len(df) >= 20 else 0
        perf_3m = ((df['Close'].iloc[-1] - df['Close'].iloc[-60]) / df['Close'].iloc[-60]) * 100 if len(df) >= 60 else 0
        
        # Volatility
        returns = df['Close'].pct_change()
        volatility = returns.std() * (252 ** 0.5) * 100  # Annualized
        
        # Get company name safely
        name = info.get('longName', ticker) if info else ticker
        
        return {
            'ticker': ticker,
            'name': name,
            'price': round(current_price, 2),
            'alignment': alignment,
            'analyst': analyst_rating,
            'technical': tech_signal,
            'score': score,
            'rsi': round(latest['RSI'], 1),
            'adx': round(latest['ADX'], 1),
            'ma20': round(latest['MA20'], 2),
            'ma50': round(latest['MA50'], 2),
            'support': support,
            'target': info.get('targetMeanPrice', 0),
            'upside': round(((info.get('targetMeanPrice', current_price) - current_price) / current_price) * 100, 1) if info.get('targetMeanPrice') else 0,
            'perf_1m': round(perf_1m, 1),
            'perf_3m': round(perf_3m, 1),
            'volatility': round(volatility, 1),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'revenue_growth': round(info.get('revenueGrowth', 0) * 100, 1) if info.get('revenueGrowth') else 0,
            'earnings_growth': round(info.get('earningsGrowth', 0) * 100, 1) if info.get('earningsGrowth') else 0,
            'analysts': info.get('numberOfAnalystOpinions', 0),
            'sector': info.get('sector', 'N/A'),
            'df': df
        }
    except Exception as e:
        return None

def create_comparison_chart(stocks_data):
    """Create normalized price comparison chart"""
    fig = go.Figure()
    
    for data in stocks_data:
        df = data['df'].copy()
        # Normalize to 100
        df['Normalized'] = (df['Close'] / df['Close'].iloc[0]) * 100
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Normalized'],
            mode='lines',
            name=data['ticker'],
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Relative Performance (Normalized to 100)',
        xaxis_title='Date',
        yaxis_title='Normalized Price',
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_metrics_comparison(stocks_data):
    """Create comparison table of key metrics"""
    comparison = []
    
    for data in stocks_data:
        comparison.append({
            'Ticker': data['ticker'],
            'Price': f"${data['price']:.2f}",
            'Alignment': data['alignment'],
            'Analyst': data['analyst'],
            'Technical': data['technical'],
            'Score': f"{data['score']}/5",
            'RSI': f"{data['rsi']:.1f}",
            'ADX': f"{data['adx']:.1f}",
            'Support': f"${data['support']:.2f}",
            'Target': f"${data['target']:.2f}" if data['target'] else 'N/A',
            'Upside': f"{data['upside']:+.1f}%",
            '1M Perf': f"{data['perf_1m']:+.1f}%",
            '3M Perf': f"{data['perf_3m']:+.1f}%",
            'Volatility': f"{data['volatility']:.1f}%",
            'Analysts': data['analysts'],
            'Rev Growth': f"{data['revenue_growth']:+.1f}%"
        })
    
    return pd.DataFrame(comparison)

def main():
    st.title("Stock Comparison")
    st.markdown("Compare multiple stocks side-by-side to find the best opportunity")
    
    # Ticker input
    st.sidebar.header("Select Stocks to Compare")
    
    # Pre-defined comparisons
    comparison_sets = {
        'CRISPR Stocks': ['NTLA', 'EDIT', 'CRSP'],
        'Large Biotech': ['VRTX', 'REGN', 'ALNY', 'GILD'],
        'Small Biotech': ['SANA', 'IONS', 'BEAM', 'ARWR'],
        'Tech Giants': ['AAPL', 'MSFT', 'NVDA', 'GOOGL'],
        'Custom': []
    }
    
    selected_set = st.sidebar.selectbox(
        "Quick Select",
        list(comparison_sets.keys())
    )
    
    if selected_set == 'Custom':
        ticker_input = st.sidebar.text_input(
            "Enter tickers (comma-separated)",
            value="NTLA,EDIT,CRSP",
            help="Enter 2-5 tickers separated by commas"
        )
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    else:
        tickers = comparison_sets[selected_set]
        st.sidebar.info(f"Comparing: {', '.join(tickers)}")
    
    if st.sidebar.button("Compare Stocks", type="primary"):
        if len(tickers) < 2:
            st.error("Please enter at least 2 tickers")
            return
        
        if len(tickers) > 5:
            st.warning("Comparing more than 5 stocks may be slow. Using first 5.")
            tickers = tickers[:5]
        
        # Load data
        st.info(f"Loading data for {', '.join(tickers)}...")
        stocks_data = []
        failed_tickers = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(tickers):
            status_text.text(f"Loading {ticker}... ({i+1}/{len(tickers)})")
            try:
                data = analyze_stock_for_comparison(ticker)
                if data:
                    stocks_data.append(data)
                    st.success(f"✓ Loaded {ticker}", icon="✅")
                else:
                    failed_tickers.append(ticker)
                    st.warning(f"✗ Could not load {ticker} - insufficient data")
            except Exception as e:
                failed_tickers.append(ticker)
                st.error(f"✗ Error loading {ticker}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(tickers))
        
        progress_bar.empty()
        status_text.empty()
        
        if not stocks_data:
            st.error("Could not load data for any stocks")
            if failed_tickers:
                st.error(f"Failed tickers: {', '.join(failed_tickers)}")
                st.info("Tips:\n- Check ticker symbols are correct\n- Try using different tickers\n- Some tickers may have insufficient trading history")
            return
        
        if len(stocks_data) < len(tickers):
            st.warning(f"Could only load {len(stocks_data)} out of {len(tickers)} stocks")
            if failed_tickers:
                st.warning(f"Failed: {', '.join(failed_tickers)}")
        
        # Store in session state
        st.session_state['comparison_data'] = stocks_data
        st.success(f"Comparing {len(stocks_data)} stocks")
    
    # Display comparison
    if 'comparison_data' in st.session_state:
        stocks_data = st.session_state['comparison_data']
        
        # Quick summary cards
        st.markdown("---")
        st.subheader("Quick Summary")
        
        cols = st.columns(len(stocks_data))
        for i, data in enumerate(stocks_data):
            with cols[i]:
                st.markdown(f"### {data['ticker']}")
                st.metric("Price", f"${data['price']:.2f}")
                
                # Alignment badge
                if data['alignment'] == 'ALIGNED-BUY':
                    st.success(f"**{data['alignment']}**")
                elif 'MIXED' in data['alignment']:
                    st.warning(f"**{data['alignment']}**")
                else:
                    st.info(f"**{data['alignment']}**")
                
                st.caption(f"Score: {data['score']}/5")
                st.caption(f"Upside: {data['upside']:+.1f}%")
        
        # Detailed comparison table
        st.markdown("---")
        st.subheader("Detailed Comparison")
        
        comparison_df = create_metrics_comparison(stocks_data)
        st.dataframe(comparison_df, width='stretch')
        
        # Price performance chart
        st.markdown("---")
        st.subheader("Price Performance")
        
        price_chart = create_comparison_chart(stocks_data)
        st.plotly_chart(price_chart, width='stretch')
        
        # Side-by-side analysis
        st.markdown("---")
        st.subheader("Detailed Analysis")
        
        tabs = st.tabs([data['ticker'] for data in stocks_data])
        
        for i, (tab, data) in enumerate(zip(tabs, stocks_data)):
            with tab:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Fundamentals**")
                    st.write(f"• Name: {data['name'][:30]}...")
                    st.write(f"• Sector: {data['sector']}")
                    st.write(f"• Market Cap: ${data['market_cap']/1e9:.1f}B" if data['market_cap'] else "• Market Cap: N/A")
                    st.write(f"• P/E Ratio: {data['pe_ratio']:.1f}" if data['pe_ratio'] else "• P/E: N/A")
                    st.write(f"• Revenue Growth: {data['revenue_growth']:+.1f}%")
                    st.write(f"• Earnings Growth: {data['earnings_growth']:+.1f}%")
                    st.write(f"• Analysts: {data['analysts']}")
                
                with col2:
                    st.markdown("**Technical**")
                    st.write(f"• Signal: {data['technical']} ({data['score']}/5)")
                    st.write(f"• RSI: {data['rsi']:.1f}")
                    st.write(f"• ADX: {data['adx']:.1f}")
                    st.write(f"• MA20: ${data['ma20']:.2f}")
                    st.write(f"• MA50: ${data['ma50']:.2f}")
                    st.write(f"• Support: ${data['support']:.2f}")
                    st.write(f"• Volatility: {data['volatility']:.1f}%")
                
                with col3:
                    st.markdown("**Entry Strategy**")
                    st.write(f"• Current: ${data['price']:.2f}")
                    st.write(f"• Target: ${data['target']:.2f}" if data['target'] else "• Target: N/A")
                    st.write(f"• Upside: {data['upside']:+.1f}%")
                    st.write(f"• 1M Perf: {data['perf_1m']:+.1f}%")
                    st.write(f"• 3M Perf: {data['perf_3m']:+.1f}%")
                    
                    if data['alignment'] == 'ALIGNED-BUY':
                        st.success("BUY NOW - Both aligned")
                    elif data['alignment'] == 'MIXED-WAIT':
                        st.warning(f"WAIT for ${data['support']:.2f}")
                    else:
                        st.info("Monitor for changes")
        
        # Recommendation
        st.markdown("---")
        st.subheader("Recommendation")
        
        # Find best opportunity
        aligned_buy = [s for s in stocks_data if s['alignment'] == 'ALIGNED-BUY']
        mixed_wait = [s for s in stocks_data if s['alignment'] == 'MIXED-WAIT']
        
        if aligned_buy:
            best = max(aligned_buy, key=lambda x: x['upside'])
            st.success(f"**Best Opportunity: {best['ticker']}**")
            st.write(f"• Status: {best['alignment']}")
            st.write(f"• Entry: ${best['price']:.2f} (NOW)")
            st.write(f"• Target: ${best['target']:.2f}")
            st.write(f"• Upside: {best['upside']:+.1f}%")
            st.write(f"• Why: Both fundamentals and technicals aligned")
        elif mixed_wait:
            best = max(mixed_wait, key=lambda x: x['upside'])
            st.warning(f"**Best Setup (Wait for Entry): {best['ticker']}**")
            st.write(f"• Status: {best['alignment']}")
            st.write(f"• Current: ${best['price']:.2f}")
            st.write(f"• Wait for: ${best['support']:.2f}")
            st.write(f"• Target: ${best['target']:.2f}")
            st.write(f"• Upside from support: {((best['target'] - best['support']) / best['support'] * 100):+.1f}%")
            st.write(f"• Why: Strong fundamentals, waiting for technical confirmation")
        else:
            st.info("No clear buy signals at this time. Monitor for changes.")
        
        # Comparison summary
        st.markdown("**Quick Comparison:**")
        for data in sorted(stocks_data, key=lambda x: x['upside'], reverse=True):
            if data['alignment'] == 'ALIGNED-BUY':
                st.write(f"✓ **{data['ticker']}**: {data['upside']:+.1f}% upside - BUY NOW")
            elif data['alignment'] == 'MIXED-WAIT':
                st.write(f"⚠ **{data['ticker']}**: {data['upside']:+.1f}% upside - WAIT for ${data['support']:.2f}")
            else:
                st.write(f"− **{data['ticker']}**: {data['upside']:+.1f}% upside - {data['alignment']}")

if __name__ == "__main__":
    main()

