"""
Watchlist - Save and monitor your favorite stocks
Track entry points and get alerts when conditions are met
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime
from advanced_indicators import TechnicalIndicators

WATCHLIST_FILE = 'watchlist.json'

def load_watchlist():
    """Load watchlist from file"""
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_watchlist(watchlist):
    """Save watchlist to file"""
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(watchlist, f, indent=2)

def get_stock_status(ticker):
    """Get current status of a stock"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period='3mo', interval='1d')
        
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
        prev = df.iloc[-2]
        
        current_price = latest['Close']
        price_change = ((current_price - prev['Close']) / prev['Close']) * 100
        
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
        
        if conditions_met >= 4:
            tech_signal = "BUY"
        elif conditions_met >= 3:
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
        else:
            alignment = "NEUTRAL"
        
        # Support calculation
        recent_low = df['Low'].iloc[-20:].min()
        support = round(recent_low * 0.99, 2)
        
        return {
            'price': round(current_price, 2),
            'change': round(price_change, 2),
            'alignment': alignment,
            'analyst': analyst_rating,
            'technical': tech_signal,
            'score': f"{conditions_met}/5",
            'rsi': round(latest['RSI'], 1),
            'ma50': round(latest['MA50'], 2),
            'support': support,
            'target': info.get('targetMeanPrice', 0),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
    except Exception as e:
        return None

def check_alerts(watchlist):
    """Check if any watchlist items meet alert conditions"""
    alerts = []
    
    for ticker, data in watchlist.items():
        status = get_stock_status(ticker)
        if not status:
            continue
        
        # Check if signal changed
        if 'last_signal' in data and data['last_signal'] != status['alignment']:
            alerts.append({
                'ticker': ticker,
                'type': 'Signal Change',
                'message': f"{ticker} signal changed: {data['last_signal']} → {status['alignment']}",
                'action': 'Review entry strategy'
            })
        
        # Check if price hit target entry
        if 'target_entry' in data and data['target_entry']:
            if status['price'] <= data['target_entry']:
                alerts.append({
                    'ticker': ticker,
                    'type': 'Price Alert',
                    'message': f"{ticker} hit target entry: ${status['price']} <= ${data['target_entry']}",
                    'action': 'Consider buying'
                })
        
        # Check if RSI in target range
        if 'rsi_target' in data and data['rsi_target']:
            rsi_min, rsi_max = data['rsi_target']
            if rsi_min <= status['rsi'] <= rsi_max:
                alerts.append({
                    'ticker': ticker,
                    'type': 'RSI Alert',
                    'message': f"{ticker} RSI in target range: {status['rsi']:.1f}",
                    'action': 'Check other conditions'
                })
        
        # Update last signal
        watchlist[ticker]['last_signal'] = status['alignment']
    
    save_watchlist(watchlist)
    return alerts

def main():
    st.title("Watchlist")
    st.markdown("Monitor your favorite stocks and get alerts")
    
    # Load watchlist
    watchlist = load_watchlist()
    
    # Add new stock
    with st.expander("Add Stock to Watchlist", expanded=len(watchlist) == 0):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            new_ticker = st.text_input("Ticker Symbol", key="new_ticker").upper()
        
        with col2:
            st.write("")
            st.write("")
            if st.button("Add to Watchlist"):
                if new_ticker and new_ticker not in watchlist:
                    status = get_stock_status(new_ticker)
                    if status:
                        watchlist[new_ticker] = {
                            'added': datetime.now().strftime('%Y-%m-%d'),
                            'notes': '',
                            'target_entry': None,
                            'rsi_target': None,
                            'last_signal': status['alignment']
                        }
                        save_watchlist(watchlist)
                        st.success(f"Added {new_ticker} to watchlist")
                        st.rerun()
                    else:
                        st.error(f"Could not fetch data for {new_ticker}")
                elif new_ticker in watchlist:
                    st.warning(f"{new_ticker} is already in watchlist")
    
    # Check for alerts
    if watchlist and st.button("Check for Alerts", type="primary"):
        with st.spinner("Checking alerts..."):
            alerts = check_alerts(watchlist)
        
        if alerts:
            st.success(f"Found {len(alerts)} alerts!")
            for alert in alerts:
                if alert['type'] == 'Signal Change':
                    st.warning(f"**{alert['ticker']}**: {alert['message']}")
                elif alert['type'] == 'Price Alert':
                    st.success(f"**{alert['ticker']}**: {alert['message']}")
                elif alert['type'] == 'RSI Alert':
                    st.info(f"**{alert['ticker']}**: {alert['message']}")
                st.caption(f"→ {alert['action']}")
        else:
            st.info("No alerts at this time")
    
    # Display watchlist
    if watchlist:
        st.markdown("---")
        st.subheader(f"Your Watchlist ({len(watchlist)} stocks)")
        
        for ticker in list(watchlist.keys()):
            with st.expander(f"**{ticker}**", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    # Get current status
                    with st.spinner(f"Loading {ticker}..."):
                        status = get_stock_status(ticker)
                    
                    if status:
                        # Display current status
                        st.metric("Current Price", f"${status['price']}", f"{status['change']:+.2f}%")
                        
                        # Alignment badge
                        if status['alignment'] == 'ALIGNED-BUY':
                            st.success(f"**{status['alignment']}**")
                        elif 'MIXED' in status['alignment']:
                            st.warning(f"**{status['alignment']}**")
                        else:
                            st.info(f"**{status['alignment']}**")
                        
                        st.caption(f"Analyst: {status['analyst']} | Technical: {status['technical']} ({status['score']})")
                        st.caption(f"RSI: {status['rsi']:.1f} | Support: ${status['support']}")
                        
                        if status['target']:
                            upside = ((status['target'] - status['price']) / status['price']) * 100
                            st.caption(f"Target: ${status['target']:.2f} ({upside:+.1f}%)")
                    else:
                        st.error(f"Could not load data for {ticker}")
                
                with col2:
                    # Alert settings
                    st.markdown("**Alert Settings:**")
                    
                    target_entry = st.number_input(
                        "Target Entry Price",
                        value=watchlist[ticker].get('target_entry', 0.0) or 0.0,
                        step=0.50,
                        key=f"entry_{ticker}"
                    )
                    
                    rsi_col1, rsi_col2 = st.columns(2)
                    with rsi_col1:
                        rsi_min = st.number_input(
                            "RSI Min",
                            0, 100, 
                            value=watchlist[ticker].get('rsi_target', [30, 50])[0] if watchlist[ticker].get('rsi_target') else 30,
                            key=f"rsi_min_{ticker}"
                        )
                    with rsi_col2:
                        rsi_max = st.number_input(
                            "RSI Max",
                            0, 100,
                            value=watchlist[ticker].get('rsi_target', [30, 50])[1] if watchlist[ticker].get('rsi_target') else 50,
                            key=f"rsi_max_{ticker}"
                        )
                    
                    notes = st.text_area(
                        "Notes",
                        value=watchlist[ticker].get('notes', ''),
                        key=f"notes_{ticker}",
                        height=80
                    )
                    
                    if st.button("Save Settings", key=f"save_{ticker}"):
                        watchlist[ticker]['target_entry'] = target_entry if target_entry > 0 else None
                        watchlist[ticker]['rsi_target'] = [rsi_min, rsi_max]
                        watchlist[ticker]['notes'] = notes
                        save_watchlist(watchlist)
                        st.success("Settings saved!")
                
                with col3:
                    st.write("")
                    st.write("")
                    st.write("")
                    if st.button("Remove", key=f"remove_{ticker}", type="secondary"):
                        del watchlist[ticker]
                        save_watchlist(watchlist)
                        st.success(f"Removed {ticker}")
                        st.rerun()
                    
                    if st.button("Analyze", key=f"analyze_{ticker}"):
                        st.session_state['current_ticker'] = ticker
                        st.info(f"Switch to main dashboard to analyze {ticker}")
    else:
        st.info("Your watchlist is empty. Add stocks to get started!")
    
    # Quick stats
    if watchlist:
        st.markdown("---")
        st.subheader("Watchlist Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total = len(watchlist)
            st.metric("Total Stocks", total)
        
        with col2:
            # Count by alignment (would need to load all statuses)
            st.metric("Added This Week", 0)  # Placeholder
        
        with col3:
            st.metric("Active Alerts", 0)  # Placeholder
        
        with col4:
            st.metric("Watchlist Age", f"{len(watchlist)} days")  # Placeholder

if __name__ == "__main__":
    main()

