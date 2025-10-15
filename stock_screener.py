"""
Stock Screener - Filter stocks by multiple criteria
Scan your biotech watchlist for the best opportunities
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from advanced_indicators import TechnicalIndicators

# Your biotech tickers
DEFAULT_TICKERS = [
    'ILMN', 'VRTX', 'REGN', 'ALNY', 'GILD', 'BIIB', 'BNTX', 'MRNA', 'CRSP', 
    'NTLA', 'BEAM', 'IONS', 'EXEL', 'ARWR', 'AXSM', 'TWST', 'AUPH', 'CLDX', 
    'SGEN', 'BCEL', 'RARE', 'ACAD', 'SRPT', 'EDIT', 'ARCT', 'DVAX', 'VMAR', 
    'RGNX', 'IMMU', 'BLUE', 'INSM', 'SRRA', 'AVRO', 'CGEN', 'SANA'
]

@st.cache_data(ttl=600)
def scan_stock(ticker):
    """Scan a single stock and return key metrics"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period='6mo', interval='1d')
        
        if df.empty or len(df) < 50:
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
        
        # Trading signal calculation
        conditions_met = 0
        total_conditions = 5
        
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
        
        # Technical signal
        if score >= 4:
            tech_signal = "BUY"
        elif score >= 3:
            tech_signal = "NEUTRAL"
        else:
            tech_signal = "SELL"
        
        # Analyst rating
        analyst_rating = info.get('recommendationKey', 'N/A').upper() if info.get('recommendationKey') else 'N/A'
        
        # Determine alignment
        analyst_bullish = analyst_rating in ['STRONG_BUY', 'BUY']
        analyst_bearish = analyst_rating in ['STRONG_SELL', 'SELL']
        tech_bullish = tech_signal in ["BUY"]
        tech_bearish = tech_signal in ["SELL"]
        
        if analyst_bullish and tech_bullish:
            alignment = "ALIGNED-BUY"
        elif analyst_bullish and tech_bearish:
            alignment = "MIXED-WAIT"
        elif analyst_bearish and tech_bullish:
            alignment = "MIXED-CAUTION"
        elif analyst_bearish and tech_bearish:
            alignment = "ALIGNED-AVOID"
        else:
            alignment = "NEUTRAL"
        
        # Calculate support level
        recent_low = df['Low'].iloc[-20:].min()
        support = round(recent_low * 0.99, 2)
        
        # Get growth metrics
        revenue_growth = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
        earnings_growth = info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0
        
        return {
            'Ticker': ticker,
            'Price': round(current_price, 2),
            'Alignment': alignment,
            'Analyst': analyst_rating,
            'Technical': tech_signal,
            'Score': f"{score}/5",
            'RSI': round(latest['RSI'], 1),
            'ADX': round(latest['ADX'], 1),
            'Support': support,
            'Target': info.get('targetMeanPrice', 0),
            'Upside': round(((info.get('targetMeanPrice', current_price) - current_price) / current_price) * 100, 1) if info.get('targetMeanPrice') else 0,
            'Rev Growth': round(revenue_growth, 1),
            'Earnings Growth': round(earnings_growth, 1),
            'Analysts': info.get('numberOfAnalystOpinions', 0),
            'Market Cap': info.get('marketCap', 0),
            'Sector': info.get('sector', 'N/A')
        }
    except Exception as e:
        return None

def run_screener(tickers, filters):
    """Run screener on list of tickers with filters"""
    st.info(f"Scanning {len(tickers)} stocks... This may take a minute.")
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Scanning {ticker}... ({i+1}/{len(tickers)})")
        result = scan_stock(ticker)
        if result:
            results.append(result)
        progress_bar.progress((i + 1) / len(tickers))
        time.sleep(0.1)  # Rate limiting
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        st.warning("No stocks found matching criteria")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Apply filters
    filtered_df = df.copy()
    
    if filters['alignment'] != 'All':
        filtered_df = filtered_df[filtered_df['Alignment'] == filters['alignment']]
    
    if filters['analyst_rating'] != 'All':
        filtered_df = filtered_df[filtered_df['Analyst'] == filters['analyst_rating']]
    
    if filters['technical_signal'] != 'All':
        filtered_df = filtered_df[filtered_df['Technical'] == filters['technical_signal']]
    
    if filters['min_rsi']:
        filtered_df = filtered_df[filtered_df['RSI'] >= filters['min_rsi']]
    
    if filters['max_rsi']:
        filtered_df = filtered_df[filtered_df['RSI'] <= filters['max_rsi']]
    
    if filters['min_adx']:
        filtered_df = filtered_df[filtered_df['ADX'] >= filters['min_adx']]
    
    if filters['min_upside']:
        filtered_df = filtered_df[filtered_df['Upside'] >= filters['min_upside']]
    
    if filters['min_rev_growth']:
        filtered_df = filtered_df[filtered_df['Rev Growth'] >= filters['min_rev_growth']]
    
    if filters['min_analysts']:
        filtered_df = filtered_df[filtered_df['Analysts'] >= filters['min_analysts']]
    
    return filtered_df

def main():
    st.title("Stock Screener")
    st.markdown("Filter your watchlist to find the best opportunities")
    
    # Sidebar filters
    st.sidebar.header("Screening Criteria")
    
    # Ticker input
    ticker_input = st.sidebar.text_area(
        "Tickers (one per line)",
        value='\n'.join(DEFAULT_TICKERS[:20]),
        height=200,
        help="Enter stock tickers, one per line"
    )
    
    tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    
    # Alignment filter
    alignment = st.sidebar.selectbox(
        "Alignment",
        ['All', 'ALIGNED-BUY', 'MIXED-WAIT', 'MIXED-CAUTION', 'ALIGNED-AVOID', 'NEUTRAL']
    )
    
    # Analyst rating filter
    analyst_rating = st.sidebar.selectbox(
        "Analyst Rating",
        ['All', 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']
    )
    
    # Technical signal filter
    technical_signal = st.sidebar.selectbox(
        "Technical Signal",
        ['All', 'BUY', 'NEUTRAL', 'SELL']
    )
    
    # RSI range
    st.sidebar.markdown("**RSI Range**")
    col1, col2 = st.sidebar.columns(2)
    min_rsi = col1.number_input("Min", 0, 100, 0)
    max_rsi = col2.number_input("Max", 0, 100, 100)
    
    # ADX minimum
    min_adx = st.sidebar.number_input(
        "Min ADX (trend strength)",
        0.0, 100.0, 0.0,
        help="0=no filter, 25+=strong trend"
    )
    
    # Upside minimum
    min_upside = st.sidebar.number_input(
        "Min Upside %",
        0.0, 100.0, 0.0,
        help="Minimum upside to analyst target"
    )
    
    # Revenue growth
    min_rev_growth = st.sidebar.number_input(
        "Min Revenue Growth %",
        -100.0, 500.0, 0.0,
        help="Year-over-year revenue growth"
    )
    
    # Analyst coverage
    min_analysts = st.sidebar.number_input(
        "Min Analyst Coverage",
        0, 100, 0,
        help="Minimum number of analysts covering stock"
    )
    
    # Scan button
    if st.sidebar.button("Run Screener", type="primary"):
        filters = {
            'alignment': alignment,
            'analyst_rating': analyst_rating,
            'technical_signal': technical_signal,
            'min_rsi': min_rsi if min_rsi > 0 else None,
            'max_rsi': max_rsi if max_rsi < 100 else None,
            'min_adx': min_adx if min_adx > 0 else None,
            'min_upside': min_upside if min_upside > 0 else None,
            'min_rev_growth': min_rev_growth if min_rev_growth != 0 else None,
            'min_analysts': min_analysts if min_analysts > 0 else None
        }
        
        results_df = run_screener(tickers, filters)
        
        if not results_df.empty:
            st.success(f"Found {len(results_df)} stocks matching criteria")
            
            # Display results
            st.dataframe(
                results_df.style.applymap(
                    lambda x: 'background-color: #90EE90' if x == 'ALIGNED-BUY' else '',
                    subset=['Alignment']
                ).applymap(
                    lambda x: 'background-color: #FFB347' if 'MIXED' in str(x) else '',
                    subset=['Alignment']
                ).applymap(
                    lambda x: 'background-color: #FFB6C1' if x == 'ALIGNED-AVOID' else '',
                    subset=['Alignment']
                ).format({
                    'Price': '${:.2f}',
                    'Support': '${:.2f}',
                    'Target': '${:.2f}',
                    'Upside': '{:+.1f}%',
                    'RSI': '{:.1f}',
                    'ADX': '{:.1f}',
                    'Rev Growth': '{:+.1f}%',
                    'Earnings Growth': '{:+.1f}%',
                    'Market Cap': '${:,.0f}'
                }),
                width='stretch'
            )
            
            # Summary stats
            st.markdown("---")
            st.subheader("Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                aligned_buy = len(results_df[results_df['Alignment'] == 'ALIGNED-BUY'])
                st.metric("ALIGNED - BUY", aligned_buy)
            
            with col2:
                mixed_wait = len(results_df[results_df['Alignment'] == 'MIXED-WAIT'])
                st.metric("MIXED - WAIT", mixed_wait)
            
            with col3:
                avg_upside = results_df['Upside'].mean()
                st.metric("Avg Upside", f"{avg_upside:+.1f}%")
            
            with col4:
                avg_rsi = results_df['RSI'].mean()
                st.metric("Avg RSI", f"{avg_rsi:.1f}")
            
            # Store results in session state
            st.session_state['screener_results'] = results_df
            
        else:
            st.warning("No stocks match your criteria. Try relaxing filters.")
    
    # Show saved results if available
    elif 'screener_results' in st.session_state:
        st.info(f"Showing previous scan results ({len(st.session_state['screener_results'])} stocks)")
        st.dataframe(st.session_state['screener_results'], width='stretch')

if __name__ == "__main__":
    main()

