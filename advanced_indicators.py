"""
Advanced Technical Indicators & News Integration
Multiple indicators beyond moving averages + integrated news sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print(" " * 25 + "ADVANCED TECHNICAL ANALYSIS & NEWS")
print("=" * 100)

# =============================================================================
# TECHNICAL INDICATORS LIBRARY
# =============================================================================

class TechnicalIndicators:
    """Calculate various technical indicators"""
    
    @staticmethod
    def RSI(data, period=14):
        """
        Relative Strength Index (RSI)
        - Above 70 = Overbought (potential sell)
        - Below 30 = Oversold (potential buy)
        - 50 = Neutral
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def MACD(data, fast=12, slow=26, signal=9):
        """
        Moving Average Convergence Divergence (MACD)
        - MACD line crosses above signal = Buy
        - MACD line crosses below signal = Sell
        """
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def BollingerBands(data, period=20, std_dev=2):
        """
        Bollinger Bands
        - Price touches upper band = Overbought
        - Price touches lower band = Oversold
        - Squeeze (bands narrow) = Low volatility, breakout coming
        """
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def Stochastic(high, low, close, period=14):
        """
        Stochastic Oscillator
        - Above 80 = Overbought
        - Below 20 = Oversold
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        stoch = 100 * (close - lowest_low) / (highest_high - lowest_low)
        return stoch
    
    @staticmethod
    def ATR(high, low, close, period=14):
        """
        Average True Range (ATR)
        - Measures volatility
        - Higher ATR = More volatile
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def OBV(close, volume):
        """
        On-Balance Volume (OBV)
        - Rising OBV = Buying pressure
        - Falling OBV = Selling pressure
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def ADX(high, low, close, period=14):
        """
        Average Directional Index (ADX)
        - Above 25 = Strong trend
        - Below 20 = Weak/no trend
        - Doesn't show direction, just strength
        """
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = pd.concat([high - low, 
                       abs(high - close.shift()), 
                       abs(low - close.shift())], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (abs(minus_dm).rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di

# =============================================================================
# ANALYZE WITH MULTIPLE INDICATORS
# =============================================================================

def comprehensive_technical_analysis(ticker, period='6mo'):
    """Run full technical analysis on a stock"""
    
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE TECHNICAL ANALYSIS: {ticker}")
    print(f"{'='*100}\n")
    
    # Download data
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval='1d')
    
    if len(df) < 50:
        print(f"Not enough data for {ticker}")
        return None
    
    # Calculate all indicators
    df['RSI'] = TechnicalIndicators.RSI(df['Close'])
    df['MACD'], df['Signal'], df['MACD_Hist'] = TechnicalIndicators.MACD(df['Close'])
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = TechnicalIndicators.BollingerBands(df['Close'])
    df['Stochastic'] = TechnicalIndicators.Stochastic(df['High'], df['Low'], df['Close'])
    df['ATR'] = TechnicalIndicators.ATR(df['High'], df['Low'], df['Close'])
    df['OBV'] = TechnicalIndicators.OBV(df['Close'], df['Volume'])
    df['ADX'], df['Plus_DI'], df['Minus_DI'] = TechnicalIndicators.ADX(df['High'], df['Low'], df['Close'])
    
    # Moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean() if len(df) >= 200 else None
    
    # Current values
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Display results
    print(f"Current Price: ${current['Close']:.2f}")
    print(f"Date: {df.index[-1].strftime('%Y-%m-%d')}")
    print("\n" + "-"*100)
    print("INDICATOR READINGS:")
    print("-"*100)
    
    # RSI Analysis
    rsi = current['RSI']
    print(f"\n1. RSI (14-day): {rsi:.2f}")
    if rsi > 70:
        print(f"   >>> OVERBOUGHT - Consider selling or taking profits")
    elif rsi < 30:
        print(f"   >>> OVERSOLD - Potential buying opportunity")
    elif rsi > 50:
        print(f"   >>> Bullish momentum")
    else:
        print(f"   >>> Bearish momentum")
    
    # MACD Analysis
    macd = current['MACD']
    signal = current['Signal']
    hist = current['MACD_Hist']
    print(f"\n2. MACD: {macd:.4f} | Signal: {signal:.4f} | Histogram: {hist:.4f}")
    if macd > signal and prev['MACD'] <= prev['Signal']:
        print(f"   >>> BULLISH CROSSOVER - Buy signal!")
    elif macd < signal and prev['MACD'] >= prev['Signal']:
        print(f"   >>> BEARISH CROSSOVER - Sell signal!")
    elif macd > signal:
        print(f"   >>> Bullish trend (MACD above signal)")
    else:
        print(f"   >>> Bearish trend (MACD below signal)")
    
    # Bollinger Bands
    price = current['Close']
    bb_upper = current['BB_Upper']
    bb_middle = current['BB_Middle']
    bb_lower = current['BB_Lower']
    bb_position = (price - bb_lower) / (bb_upper - bb_lower) * 100
    
    print(f"\n3. Bollinger Bands:")
    print(f"   Upper: ${bb_upper:.2f} | Middle: ${bb_middle:.2f} | Lower: ${bb_lower:.2f}")
    print(f"   Price position: {bb_position:.1f}% (0%=lower band, 100%=upper band)")
    if price > bb_upper:
        print(f"   >>> Above upper band - OVERBOUGHT")
    elif price < bb_lower:
        print(f"   >>> Below lower band - OVERSOLD")
    elif bb_position > 70:
        print(f"   >>> Near upper band - Approaching overbought")
    elif bb_position < 30:
        print(f"   >>> Near lower band - Approaching oversold")
    else:
        print(f"   >>> In the middle - Neutral zone")
    
    # Stochastic
    stoch = current['Stochastic']
    print(f"\n4. Stochastic Oscillator: {stoch:.2f}")
    if stoch > 80:
        print(f"   >>> OVERBOUGHT - Reversal possible")
    elif stoch < 20:
        print(f"   >>> OVERSOLD - Bounce possible")
    else:
        print(f"   >>> Neutral")
    
    # ATR (Volatility)
    atr = current['ATR']
    atr_pct = (atr / price) * 100
    print(f"\n5. ATR (Volatility): ${atr:.2f} ({atr_pct:.2f}% of price)")
    if atr_pct > 5:
        print(f"   >>> HIGH VOLATILITY - Expect large price swings")
    elif atr_pct > 3:
        print(f"   >>> Moderate volatility")
    else:
        print(f"   >>> Low volatility - Quiet market")
    
    # OBV (Volume)
    obv = current['OBV']
    obv_change = ((obv - df['OBV'].iloc[-20]) / df['OBV'].iloc[-20]) * 100
    print(f"\n6. On-Balance Volume (OBV): {obv:,.0f}")
    print(f"   20-day change: {obv_change:+.2f}%")
    if obv_change > 10:
        print(f"   >>> Strong buying pressure")
    elif obv_change < -10:
        print(f"   >>> Strong selling pressure")
    else:
        print(f"   >>> Neutral volume trend")
    
    # ADX (Trend Strength)
    adx = current['ADX']
    plus_di = current['Plus_DI']
    minus_di = current['Minus_DI']
    print(f"\n7. ADX (Trend Strength): {adx:.2f}")
    print(f"   +DI: {plus_di:.2f} | -DI: {minus_di:.2f}")
    if adx > 25:
        if plus_di > minus_di:
            print(f"   >>> STRONG UPTREND")
        else:
            print(f"   >>> STRONG DOWNTREND")
    elif adx > 20:
        print(f"   >>> Moderate trend")
    else:
        print(f"   >>> WEAK/NO TREND - Choppy market")
    
    # Moving Averages
    ma20 = current['MA20']
    ma50 = current['MA50']
    print(f"\n8. Moving Averages:")
    print(f"   Price: ${price:.2f}")
    print(f"   MA20:  ${ma20:.2f} ({((price/ma20-1)*100):+.2f}%)")
    print(f"   MA50:  ${ma50:.2f} ({((price/ma50-1)*100):+.2f}%)")
    if not pd.isna(current['MA200']):
        ma200 = current['MA200']
        print(f"   MA200: ${ma200:.2f} ({((price/ma200-1)*100):+.2f}%)")
    
    if price > ma20 > ma50:
        print(f"   >>> BULLISH ALIGNMENT - Strong uptrend")
    elif price < ma20 < ma50:
        print(f"   >>> BEARISH ALIGNMENT - Strong downtrend")
    else:
        print(f"   >>> Mixed signals")
    
    # Overall Signal
    print("\n" + "-"*100)
    print("OVERALL SIGNAL SUMMARY:")
    print("-"*100)
    
    signals = {
        'RSI': 1 if 30 < rsi < 70 else (0.5 if rsi < 30 else -0.5),
        'MACD': 1 if macd > signal else -1,
        'BB': 0 if bb_lower < price < bb_upper else (-1 if price > bb_upper else 1),
        'Stochastic': 1 if stoch < 20 else (-1 if stoch > 80 else 0),
        'MA': 1 if price > ma20 > ma50 else (-1 if price < ma20 < ma50 else 0),
        'Trend': 1 if (adx > 25 and plus_di > minus_di) else (-1 if (adx > 25 and minus_di > plus_di) else 0),
        'Volume': 1 if obv_change > 5 else (-1 if obv_change < -5 else 0)
    }
    
    total_signal = sum(signals.values())
    max_signal = len(signals)
    
    print(f"\nSignal Score: {total_signal:.1f} / {max_signal}")
    for indicator, score in signals.items():
        emoji = "[+]" if score > 0 else ("[-]" if score < 0 else "[=]")
        print(f"  {emoji} {indicator}: {score:+.1f}")
    
    if total_signal > 3:
        print(f"\n>>> STRONG BUY SIGNAL")
    elif total_signal > 1:
        print(f"\n>>> Moderate Buy")
    elif total_signal > -1:
        print(f"\n>>> NEUTRAL / HOLD")
    elif total_signal > -3:
        print(f"\n>>> Moderate Sell")
    else:
        print(f"\n>>> STRONG SELL SIGNAL")
    
    return df

# =============================================================================
# NEWS INTEGRATION
# =============================================================================

def get_enhanced_news(ticker):
    """
    Get news from multiple sources
    """
    print(f"\n{'='*100}")
    print(f"NEWS & CATALYSTS: {ticker}")
    print(f"{'='*100}\n")
    
    stock = yf.Ticker(ticker)
    
    # Method 1: Yahoo Finance News (yfinance built-in)
    print("-"*100)
    print("METHOD 1: Yahoo Finance News Feed")
    print("-"*100)
    try:
        news = stock.news
        if news and len(news) > 0:
            for i, article in enumerate(news[:5], 1):
                title = article.get('title', 'No title')
                publisher = article.get('publisher', 'Unknown')
                
                # Fix timestamp
                timestamp = article.get('providerPublishTime', 0)
                if timestamp and timestamp > 0:
                    pub_date = datetime.fromtimestamp(timestamp)
                    print(f"\n{i}. {title}")
                    print(f"   Source: {publisher} | Date: {pub_date.strftime('%Y-%m-%d %H:%M')}")
                    
                    # Try to get summary
                    if 'summary' in article and article['summary']:
                        summary = article['summary'][:200]
                        print(f"   Summary: {summary}...")
        else:
            print("No news available via this method")
    except Exception as e:
        print(f"Error: {str(e)[:100]}")
    
    # Method 2: Company Info & Fundamentals
    print(f"\n{'-'*100}")
    print("METHOD 2: Company Fundamentals & Key Events")
    print("-"*100)
    try:
        info = stock.info
        
        print(f"\nCompany: {info.get('longName', 'N/A')}")
        print(f"Sector: {info.get('sector', 'N/A')}")
        print(f"Industry: {info.get('industry', 'N/A')}")
        
        # Earnings & Dividends (key catalysts)
        if 'earningsDate' in info and info['earningsDate']:
            earnings_dates = info['earningsDate']
            if earnings_dates:
                print(f"\n[CATALYST] UPCOMING EVENTS:")
                for date in earnings_dates:
                    if isinstance(date, (int, float)):
                        earnings_dt = datetime.fromtimestamp(date)
                        print(f"   Next Earnings: {earnings_dt.strftime('%Y-%m-%d')}")
        
        if 'exDividendDate' in info and info['exDividendDate']:
            div_date = datetime.fromtimestamp(info['exDividendDate'])
            print(f"   Ex-Dividend Date: {div_date.strftime('%Y-%m-%d')}")
        
        # Analyst recommendations
        if 'recommendationKey' in info:
            rec = info['recommendationKey']
            print(f"\n[ANALYST] Consensus: {rec.upper()}")
        
        if 'targetMeanPrice' in info and info['targetMeanPrice']:
            target = info['targetMeanPrice']
            current = info.get('currentPrice', info.get('regularMarketPrice', 0))
            if current:
                upside = ((target - current) / current) * 100
                print(f"   Analyst Price Target: ${target:.2f} ({upside:+.1f}% upside)")
        
    except Exception as e:
        print(f"Error: {str(e)[:100]}")
    
    # Method 3: SEC Filings
    print(f"\n{'-'*100}")
    print("METHOD 3: Recent SEC Filings")
    print("-"*100)
    try:
        # Get recent filings
        filings = stock.get_sec_filings()
        if filings is not None and len(filings) > 0:
            print("\nRecent filings (potential catalysts):")
            for i, filing in filings.head(5).iterrows():
                if 'type' in filing and 'date' in filing:
                    print(f"  - {filing['type']:6s}: {filing['date']} - {filing.get('title', 'N/A')[:60]}")
        else:
            print("No recent SEC filings available")
    except Exception as e:
        print(f"SEC filings not available: {str(e)[:50]}")
    
    # Method 4: Instructions for external APIs
    print(f"\n{'-'*100}")
    print("METHOD 4: How to Use External News APIs")
    print("-"*100)
    print("""
To get better news coverage, you can use these services:

1. NewsAPI (newsapi.org)
   - Free: 100 requests/day
   - Installation: pip install newsapi-python
   - Code example:
     
     from newsapi import NewsApiClient
     newsapi = NewsApiClient(api_key='YOUR_API_KEY')
     articles = newsapi.get_everything(q='{ticker}', language='en', sort_by='publishedAt')

2. Alpha Vantage News (alphavantage.co)
   - Free tier available
   - Installation: pip install alpha_vantage
   - Provides news sentiment analysis

3. Finnhub (finnhub.io)
   - Free: 60 calls/minute
   - Installation: pip install finnhub-python
   - Code example:
     
     import finnhub
     finnhub_client = finnhub.Client(api_key="YOUR_API_KEY")
     news = finnhub_client.company_news('{ticker}', _from="2025-01-01", to="2025-12-31")

4. Polygon.io
   - Your friend's API (if you get your own)
   - Best quality financial news with sentiment scores
   
NOTE: Sign up for free API keys at these services for better news coverage!
    """.format(ticker=ticker))

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    
    # Analyze top biotech performers with all indicators
    tickers = ['NTLA', 'INSM', 'IONS', 'SANA']
    
    for ticker in tickers:
        df = comprehensive_technical_analysis(ticker, period='6mo')
        get_enhanced_news(ticker)
        print("\n\n")
    
    print("="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print("""
SUMMARY - Technical Indicators Explained:

1. RSI - Momentum indicator (overbought/oversold)
2. MACD - Trend following (buy/sell crossovers)
3. Bollinger Bands - Volatility bands (price extremes)
4. Stochastic - Momentum oscillator
5. ATR - Volatility measure
6. OBV - Volume-based momentum
7. ADX - Trend strength (not direction)
8. Moving Averages - Trend direction

Use these TOGETHER for better signals. One indicator alone can be misleading!
    """)

