"""
Live Stock Trading Dashboard
Interactive web dashboard with real-time data and indicators

Run with: streamlit run live_dashboard.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from advanced_indicators import TechnicalIndicators
import portfolio_tracker as pt
from backtest_engine import BacktestEngine
from news_analyzer import NewsAnalyzer
from sector_analyzer import SectorAnalyzer
from paper_trading import PaperTradingEngine
from enhanced_valuation import EnhancedValuation
import time
import random
import concurrent.futures
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dashboard")

# Page config
st.set_page_config(
    page_title="Stock Trading Dashboard",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Wider sidebar CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 350px;
        max-width: 350px;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 20px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .buy-signal {
        color: #00A86B;
        font-weight: bold;
        font-size: 28px;
    }
    .sell-signal {
        color: #DC143C;
        font-weight: bold;
        font-size: 28px;
    }
    .neutral-signal {
        color: #FF8C00;
        font-weight: bold;
        font-size: 28px;
    }
    /* Improve expander styling */
    .streamlit-expanderHeader {
        font-size: 16px;
        font-weight: 600;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    /* Improve metric styling */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 600;
    }
    [data-testid="stMetricDelta"] {
        font-size: 14px;
    }
    /* Improve container styling */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIGNAL CALCULATION SYSTEM
# ============================================================================

@dataclass
class SignalParams:
    """Parameters for signal calculation"""
    rsi_band: tuple = (40, 60)
    adx_thr: int = 25
    vol_mult: float = 1.0
    donchian_n: int = 20
    hysteresis: int = 5
    buy_score_thr: int = 70
    sell_score_thr: int = 30
    weekly_confirm: bool = True

def _slope(series, n=3):
    """Calculate slope of last n values"""
    if series is None or len(series) < n:
        return 0.0
    try:
        y = np.array(series.iloc[-n:], dtype=float)
        x = np.arange(n, dtype=float)
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0

def _weekly(df):
    """Resample to weekly"""
    try:
        w = df.resample('W').last().dropna()
        return w if len(w) >= 10 else None
    except Exception:
        return None

def compute_signal(df: pd.DataFrame, info: dict, params: SignalParams, last_label: str = None):
    """
    Compute BUY/HOLD/SELL signal with scoring and explanations.
    
    Returns dict with:
        - label: "BUY", "HOLD", or "SELL"
        - score: 0-100
        - reasons: list of explanation strings
        - risk: dict with stop, target, atr_pct
    """
    out = {"label": "HOLD", "score": 50, "reasons": [], "risk": {}}
    
    if df is None or df.empty or len(df) < 60:
        out["reasons"].append("Insufficient data")
        return out

    try:
        cur = df.iloc[-1]
        price = float(cur['Close'])
        ma20 = float(df['MA20'].iloc[-1]) if 'MA20' in df.columns and not pd.isna(df['MA20'].iloc[-1]) else np.nan
        ma50 = float(df['MA50'].iloc[-1]) if 'MA50' in df.columns and not pd.isna(df['MA50'].iloc[-1]) else np.nan
        ma200 = float(df['MA200'].iloc[-1]) if 'MA200' in df.columns and not pd.isna(df['MA200'].iloc[-1]) else np.nan
        rsi = float(df['RSI'].iloc[-1]) if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else np.nan
        macd = float(df['MACD'].iloc[-1]) if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]) else np.nan
        macd_sig = float(df['MACD_Signal'].iloc[-1]) if 'MACD_Signal' in df.columns and not pd.isna(df['MACD_Signal'].iloc[-1]) else np.nan
        adx = float(df['ADX'].iloc[-1]) if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[-1]) else np.nan
        vol = float(cur['Volume']) if 'Volume' in cur.index else np.nan
        vol_ma = float(df['Volume_MA20'].iloc[-1]) if 'Volume_MA20' in df.columns and not pd.isna(df['Volume_MA20'].iloc[-1]) else 1.0
        atr = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns and not pd.isna(df['ATR'].iloc[-1]) else 0.0

        score = 0
        
        # 1. REGIME FILTER - Identify trend direction
        up = False
        if not np.isnan(ma200) and not np.isnan(ma50):
            up = (price > ma200) and (ma50 > ma200) and _slope(df['MA50']) > 0
        elif not np.isnan(ma50) and not np.isnan(ma20):
            up = (ma50 > ma20) and _slope(df['MA50']) > 0

        if up:
            score += 20
            out["reasons"].append("[+] Uptrend regime")
        else:
            if _slope(df['MA50']) < 0:
                score -= 15
                out["reasons"].append("[-] Downtrend regime")
            else:
                out["reasons"].append("[~] Neutral regime")

        # 2. MOMENTUM QUALITY - Prefer rising momentum
        if not np.isnan(macd) and not np.isnan(macd_sig):
            if (macd > macd_sig) and _slope(df['MACD']) > 0:
                score += 15
                out["reasons"].append("[+] MACD rising above signal")
            elif macd < macd_sig:
                score -= 5
                out["reasons"].append("[-] MACD below signal")
        
        # 3. RSI ANALYSIS - Momentum and overbought/oversold
        if not np.isnan(rsi):
            if rsi >= 80:
                score -= 15
                out["reasons"].append("[-] RSI extremely overbought (>80)")
            elif rsi >= 70:
                score -= 5
                out["reasons"].append("[!] RSI overbought (>70)")
            elif rsi <= 20:
                score += 10
                out["reasons"].append("[+] RSI oversold (<20) - potential bounce")
            elif rsi <= 30:
                score += 5
                out["reasons"].append("[+] RSI oversold zone (<30)")
            elif params.rsi_band[0] <= rsi <= params.rsi_band[1]:
                if up and _slope(df['RSI']) > 0:
                    score += 10
                    out["reasons"].append("[+] RSI in healthy zone & rising")
                else:
                    out["reasons"].append("[~] RSI in neutral zone")

        # 4. VOLUME CONFIRMATION
        if not np.isnan(vol) and vol_ma > 0:
            vol_ok = vol > params.vol_mult * vol_ma
            if vol_ok:
                score += 10
                out["reasons"].append(f"[+] Volume {vol/vol_ma:.1f}x above average")
            else:
                score -= 5
                out["reasons"].append(f"[-] Weak volume ({vol/vol_ma:.1f}x avg)")

        # 5. BREAKOUT/BREAKDOWN (Donchian)
        if len(df) >= params.donchian_n:
            h = float(df['High'].iloc[-params.donchian_n:].max())
            l = float(df['Low'].iloc[-params.donchian_n:].min())
            vol_ok = (not np.isnan(vol)) and (vol > params.vol_mult * vol_ma)
            
            if price > h and vol_ok:
                score += 15
                out["reasons"].append(f"[+] Breakout above {params.donchian_n}d high")
            elif price < l:
                score -= 15
                out["reasons"].append(f"[-] Breakdown below {params.donchian_n}d low")

        # 6. TREND STRENGTH (ADX)
        if not np.isnan(adx):
            if adx > params.adx_thr and _slope(df['ADX']) > 0:
                score += 10
                out["reasons"].append(f"[+] Strong rising trend (ADX {adx:.1f})")
            elif adx < params.adx_thr:
                score -= 5
                out["reasons"].append(f"[-] Weak trend (ADX {adx:.1f} < {params.adx_thr})")

        # 7. MULTI-TIMEFRAME CONFIRMATION (Weekly)
        if params.weekly_confirm and len(df) >= 100:
            w = _weekly(df[['Open','High','Low','Close','Volume']])
            if w is not None and len(w) >= 20:
                w_ma20 = w['Close'].rolling(20).mean()
                if len(w_ma20) > 0 and _slope(w_ma20) > 0:
                    score += 10
                    out["reasons"].append("[+] Weekly trend confirms uptrend")
                else:
                    score -= 5
                    out["reasons"].append("[-] Weekly trend not supportive")

        # 8. RISK LEVELS
        stop = float(max(cur['Low'], price - 2*atr)) if atr > 0 else float(cur['Low'])
        tgt1 = float(price + 2*atr) if atr > 0 else float(price * 1.05)
        out["risk"] = {
            "atr_pct": float(atr / price * 100) if price > 0 else 0.0,
            "stop": stop,
            "tgt1": tgt1,
            "reward_risk": float((tgt1 - price) / (price - stop)) if (price - stop) > 0 else 0.0
        }

        # 9. MAP SCORE TO LABEL WITH HYSTERESIS
        label = "HOLD"
        if score >= params.buy_score_thr:
            label = "BUY"
        elif score <= params.sell_score_thr:
            label = "SELL"

        # Apply hysteresis to prevent flip-flopping
        if last_label == "BUY" and score >= (params.buy_score_thr - params.hysteresis):
            label = "BUY"
        if last_label == "SELL" and score <= (params.sell_score_thr + params.hysteresis):
            label = "SELL"

        out["label"] = label
        out["score"] = int(max(0, min(100, score)))
        
    except Exception as e:
        log.error(f"Error in compute_signal: {e}")
        out["reasons"].append(f"Error: {str(e)}")
    
    return out

def position_size_calculator(capital, risk_pct, entry, stop):
    """Calculate position size based on risk management"""
    risk_per_share = max(entry - stop, 0.01)
    shares = int((capital * risk_pct / 100) / risk_per_share)
    return {
        'shares': shares,
        'position_value': shares * entry,
        'risk_amount': shares * risk_per_share,
        'risk_pct': risk_pct
    }

# ============================================================================
# DATA LOADING WITH RETRY AND CACHING
# ============================================================================

def _retry_with_backoff(n=3, base=0.5, jitter=0.4):
    """Decorator for retry with exponential backoff"""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            for i in range(n):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if i == n - 1:
                        log.error(f"Failed after {n} retries: {e}")
                        raise
                    wait = base * (2 ** i) + random.random() * jitter
                    log.warning(f"Retry {i+1}/{n} after {wait:.2f}s: {e}")
                    time.sleep(wait)
        return wrapper
    return decorator

@st.cache_data(ttl=300, max_entries=256, show_spinner=False)
@_retry_with_backoff(n=3)
def load_stock_data(ticker, period='6mo'):
    """Load stock data with caching and retry logic"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval='1d', timeout=10)
        
        if df is None or df.empty:
            log.warning(f"Empty data for {ticker}")
            return pd.DataFrame(), {}
        
        # Get info separately with error handling
        info = {}
        try:
            info = stock.info
        except Exception as e:
            log.warning(f"Could not fetch info for {ticker}: {e}")
        
        return df.copy(), info
    except Exception as e:
        log.error(f"Error loading {ticker}: {e}")
        return pd.DataFrame(), {}

def load_many_parallel(tickers, period='3mo'):
    """Load multiple tickers in parallel"""
    results = {}
    
    def load_one(ticker):
        try:
            df, info = load_stock_data(ticker, period)
            return ticker, df, info
        except Exception as e:
            log.error(f"Failed to load {ticker}: {e}")
            return ticker, pd.DataFrame(), {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(tickers))) as executor:
        futures = {executor.submit(load_one, t): t for t in tickers}
        
        for future in concurrent.futures.as_completed(futures):
            ticker, df, info = future.result()
            results[ticker] = (df, info)
    
    return results

@st.cache_data(ttl=60)  # Cache for 1 minute for watchlist
def get_quick_stock_data(ticker):
    """Get quick stock data for watchlist display"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period='5d', interval='1d')
        
        if df.empty or len(df) < 2:
            return None
        
        latest = df.iloc[-1]['Close']
        prev = df.iloc[-2]['Close']
        change = ((latest - prev) / prev) * 100
        
        # Quick signal calculation
        df['MA20'] = df['Close'].rolling(window=min(20, len(df))).mean()
        df['MA50'] = df['Close'].rolling(window=min(50, len(df))).mean()
        df['RSI'] = TechnicalIndicators.RSI(df['Close'])
        
        latest_row = df.iloc[-1]
        
        # Simple signal
        conditions_met = 0
        if len(df) >= 50 and latest_row['Close'] > latest_row['MA20'] > latest_row['MA50']:
            conditions_met += 1
        if 40 < latest_row['RSI'] < 60:
            conditions_met += 1
        
        macd_data = TechnicalIndicators.MACD(df['Close'])
        if macd_data['MACD'].iloc[-1] > macd_data['Signal'].iloc[-1]:
            conditions_met += 1
        
        # Smart signal logic
        rsi_val = latest_row['RSI']
        if rsi_val >= 80:
            signal = "HOLD"
        elif rsi_val >= 70 and conditions_met < 3:
            signal = "HOLD"
        elif conditions_met >= 3:
            signal = "BUY"
        elif conditions_met <= 1:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        # Get support
        support = round(df['Low'].min() * 0.99, 2)
        
        return {
            'price': round(latest, 2),
            'change': round(change, 2),
            'signal': signal,
            'support': support,
            'rsi': round(latest_row['RSI'], 1)
        }
    except:
        return None

@st.cache_data(ttl=300)
def load_earnings_data(ticker):
    """Load earnings and fundamental data"""
    stock = yf.Ticker(ticker)
    
    # Get quarterly income statement (new API)
    quarterly_income = None
    try:
        quarterly_income = stock.quarterly_income_stmt
    except:
        pass
    
    # Process earnings data from income statement
    earnings_df = None
    if quarterly_income is not None and not quarterly_income.empty:
        # Create a DataFrame with Revenue and Net Income
        data = {}
        
        if 'Total Revenue' in quarterly_income.index:
            revenue = quarterly_income.loc['Total Revenue']
            data['Revenue'] = revenue
        
        if 'Net Income' in quarterly_income.index:
            net_income = quarterly_income.loc['Net Income']
            data['Net Income'] = net_income
            
            # Calculate EPS (Net Income / Shares Outstanding)
            # Try to get shares from info
            try:
                shares = stock.info.get('sharesOutstanding', 1)
                if shares and shares > 0:
                    data['EPS'] = net_income / shares
            except:
                pass
        
        if data:
            earnings_df = pd.DataFrame(data)
            # Transpose so dates are index
            earnings_df = earnings_df.T
            # Sort by date
            earnings_df = earnings_df.sort_index(axis=1)
    
    # Get earnings dates for estimates
    earnings_dates = None
    try:
        earnings_dates = stock.earnings_dates
    except:
        pass
    
    return {
        'earnings_df': earnings_df,
        'quarterly_income': quarterly_income,
        'earnings_dates': earnings_dates
    }

def calculate_all_indicators(df):
    """Calculate all technical indicators"""
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean() if len(df) >= 200 else np.nan
    
    # RSI
    df['RSI'] = TechnicalIndicators.RSI(df['Close'])
    
    # MACD
    macd, signal, hist = TechnicalIndicators.MACD(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist
    
    # Bollinger Bands
    upper, middle, lower = TechnicalIndicators.BollingerBands(df['Close'])
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower
    
    # Volume
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['OBV'] = TechnicalIndicators.OBV(df['Close'], df['Volume'])
    
    # ATR
    df['ATR'] = TechnicalIndicators.ATR(df['High'], df['Low'], df['Close'])
    
    # Stochastic
    df['Stochastic'] = TechnicalIndicators.Stochastic(df['High'], df['Low'], df['Close'])
    
    # ADX
    adx, plus_di, minus_di = TechnicalIndicators.ADX(df['High'], df['Low'], df['Close'])
    df['ADX'] = adx
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    
    return df

def create_price_chart(df, ticker):
    """Create interactive price chart with indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=(f'{ticker} - Price Chart', 'Volume', 'RSI', 'MACD')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving Averages
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MA20'], name='MA20', 
                  line=dict(color='blue', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MA50'], name='MA50', 
                  line=dict(color='orange', width=1)),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                  line=dict(color='gray', width=1, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                  line=dict(color='gray', width=1, dash='dash'),
                  fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
        row=1, col=1
    )
    
    # Volume
    colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume',
               marker_color=colors),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                  line=dict(color='purple', width=2)),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                  line=dict(color='blue', width=2)),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                  line=dict(color='red', width=2)),
        row=4, col=1
    )
    
    # MACD Histogram
    colors_hist = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
    fig.add_trace(
        go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram',
               marker_color=colors_hist),
        row=4, col=1
    )
    
    # Layout
    fig.update_layout(
        height=1000,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    
    return fig

def create_eps_chart(earnings_data):
    """Create EPS trend chart with actual vs estimate"""
    earnings_df = earnings_data.get('earnings_df')
    
    if earnings_df is None or earnings_df.empty:
        return None
    
    # Get last 8 quarters
    df = earnings_df.iloc[:, -8:] if earnings_df.shape[1] > 8 else earnings_df
    
    fig = go.Figure()
    
    # Format quarter labels
    quarter_labels = [col.strftime('%Y-%m') if hasattr(col, 'strftime') else str(col) 
                     for col in df.columns]
    
    # EPS line
    if 'EPS' in df.index:
        eps_values = df.loc['EPS'].values
        fig.add_trace(go.Scatter(
            x=quarter_labels,
            y=eps_values,
            mode='lines+markers',
            name='EPS',
            line=dict(color='#1E88E5', width=3),
            marker=dict(size=10),
            hovertemplate='%{x}<br>EPS: $%{y:.2f}<extra></extra>'
        ))
    
    # Net Income line
    if 'Net Income' in df.index:
        net_income_billions = df.loc['Net Income'].values / 1e9
        fig.add_trace(go.Scatter(
            x=quarter_labels,
            y=net_income_billions,
            mode='lines+markers',
            name='Net Income (B)',
            line=dict(color='#43A047', width=2, dash='dash'),
            marker=dict(size=8),
            yaxis='y2',
            hovertemplate='%{x}<br>Net Income: $%{y:.2f}B<extra></extra>'
        ))
    
    fig.update_layout(
        title='Earnings Per Share & Net Income Trend',
        xaxis_title='Quarter',
        yaxis_title='EPS ($)',
        yaxis2=dict(
            title='Net Income (Billions)',
            overlaying='y',
            side='right'
        ),
        height=400,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_revenue_earnings_chart(earnings_data):
    """Create Revenue vs Earnings bar chart"""
    earnings_df = earnings_data.get('earnings_df')
    
    if earnings_df is None or earnings_df.empty:
        return None
    
    # Get last 6 quarters
    df = earnings_df.iloc[:, -6:] if earnings_df.shape[1] > 6 else earnings_df
    
    # Format quarter labels
    quarter_labels = [col.strftime('%Y-%m') if hasattr(col, 'strftime') else str(col) 
                     for col in df.columns]
    
    fig = go.Figure()
    
    # Determine if we should use billions or millions based on max values
    max_revenue = df.loc['Revenue'].max() if 'Revenue' in df.index else 0
    max_net_income = abs(df.loc['Net Income'].max()) if 'Net Income' in df.index else 0
    max_value = max(max_revenue, max_net_income)
    
    use_billions = max_value >= 1e9
    divisor = 1e9 if use_billions else 1e6
    unit_label = 'B' if use_billions else 'M'
    yaxis_title = f'Amount ({unit_label}illions $)'
    
    # Revenue bars
    if 'Revenue' in df.index:
        revenue_values = df.loc['Revenue'].values / divisor
        fig.add_trace(go.Bar(
            x=quarter_labels,
            y=revenue_values,
            name=f'Revenue ({unit_label})',
            marker_color='#64B5F6',
            yaxis='y',
            hovertemplate=f'%{{x}}<br>Revenue: $%{{y:.2f}}{unit_label}<extra></extra>'
        ))
    
    # Net Income bars
    if 'Net Income' in df.index:
        net_income_values = df.loc['Net Income'].values / divisor
        colors = ['#66BB6A' if val >= 0 else '#EF5350' for val in net_income_values]
        fig.add_trace(go.Bar(
            x=quarter_labels,
            y=net_income_values,
            name=f'Net Income ({unit_label})',
            marker_color=colors,
            yaxis='y',
            hovertemplate=f'%{{x}}<br>Net Income: $%{{y:.2f}}{unit_label}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Revenue vs. Net Income',
        xaxis_title='Quarter',
        yaxis=dict(
            title=yaxis_title,
            side='left'
        ),
        height=400,
        barmode='group',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def get_trading_signal(df):
    """Calculate trading signal with detailed explanations"""
    latest = df.iloc[-1]
    
    # Check conditions with explanations
    conditions = {
        'Trend': {
            'passed': latest['Close'] > latest['MA20'] > latest['MA50'],
            'value': f"Price ${latest['Close']:.2f} > MA20 ${latest['MA20']:.2f} > MA50 ${latest['MA50']:.2f}",
            'explanation': 'Bullish alignment' if latest['Close'] > latest['MA20'] > latest['MA50'] else 'Not aligned'
        },
        'RSI': {
            'passed': 40 < latest['RSI'] < 60,
            'value': f"{latest['RSI']:.1f}",
            'explanation': get_rsi_interpretation(latest['RSI'])
        },
        'MACD': {
            'passed': latest['MACD'] > latest['MACD_Signal'],
            'value': f"MACD {latest['MACD']:.3f} vs Signal {latest['MACD_Signal']:.3f}",
            'explanation': 'Bullish' if latest['MACD'] > latest['MACD_Signal'] else 'Bearish'
        },
        'ADX': {
            'passed': latest['ADX'] > 25,
            'value': f"{latest['ADX']:.1f}",
            'explanation': get_adx_interpretation(latest['ADX'])
        },
        'Volume': {
            'passed': latest['Volume'] > latest['Volume_MA20'],
            'value': f"{latest['Volume']:,.0f} vs Avg {latest['Volume_MA20']:,.0f}",
            'explanation': 'Above average' if latest['Volume'] > latest['Volume_MA20'] else 'Below average'
        }
    }
    
    score = sum(c['passed'] for c in conditions.values())
    
    # Determine signal with smarter logic
    # Critical disqualifiers - these override score
    rsi = latest['RSI']
    volume_weak = latest['Volume'] < latest['Volume_MA20']
    
    if rsi >= 80:
        # Extremely overbought - major red flag
        signal = "HOLD / OVERBOUGHT"
        color = "orange"
    elif rsi >= 70 and volume_weak:
        # Overbought with weak volume - caution
        signal = "HOLD / CAUTION"
        color = "orange"
    elif score >= 4 and rsi < 70:
        signal = "STRONG BUY"
        color = "green"
    elif score >= 3 and rsi < 70:
        signal = "BUY"
        color = "green"
    elif latest['Close'] < latest['MA20'] and rsi < 40:
        signal = "SELL"
        color = "red"
    elif score <= 2:
        signal = "NEUTRAL / HOLD"
        color = "orange"
    else:
        signal = "HOLD"
        color = "orange"
    
    return signal, color, score, conditions

def get_rsi_interpretation(rsi):
    """
    RSI (Relative Strength Index) Interpretation
    Real data based on momentum calculation
    """
    if rsi >= 80:
        return "EXTREMELY OVERBOUGHT - Strong sell signal"
    elif rsi >= 70:
        return "Overbought - Consider taking profits"
    elif rsi >= 60:
        return "Strong momentum - Trending up"
    elif rsi >= 50:
        return "Bullish - Above neutral"
    elif rsi >= 40:
        return "Neutral to slightly bearish"
    elif rsi >= 30:
        return "Bearish - Below neutral"
    elif rsi >= 20:
        return "Oversold - Potential buy zone"
    else:
        return "EXTREMELY OVERSOLD - Strong buy signal"

def get_adx_interpretation(adx):
    """
    ADX (Average Directional Index) Interpretation
    Real data based on trend strength calculation
    """
    if adx >= 50:
        return "VERY STRONG TREND - Momentum strategies work"
    elif adx >= 25:
        return "Strong trend - Follow the trend"
    elif adx >= 20:
        return "Moderate trend - Trend developing"
    else:
        return "WEAK/NO TREND - Use mean reversion strategies"

# Main App
def show_watchlist_view():
    """Show full watchlist page"""
    # Back button
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("← Back to Dashboard", type="primary", use_container_width=True):
            st.session_state['show_watchlist'] = False
            st.rerun()
    with col2:
        st.title("Watchlist Overview")
    
    watchlist = st.session_state.get('watchlist', {})
    
    if not watchlist:
        st.info("Your watchlist is empty. Add stocks from the main dashboard!")
        return
    
    st.success(f"Monitoring {len(watchlist)} stocks")
    
    # Load data for all watchlist stocks
    watchlist_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(watchlist.keys()):
        status_text.text(f"Loading {ticker}... ({i+1}/{len(watchlist)})")
        
        try:
            df, info = load_stock_data(ticker, period='3mo')
            
            if df.empty:
                continue
                
            df = calculate_all_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Get signal params from session state
            params = st.session_state.get('signal_params', SignalParams())
            
            # Get last signal for hysteresis
            last_lbl = watchlist.get(ticker, {}).get('_last_label') if isinstance(watchlist.get(ticker), dict) else None
            
            # Compute signal using new system
            sig = compute_signal(df, info, params, last_label=last_lbl)
            signal = sig["label"]
            score = sig["score"]
            reasons = sig["reasons"]
            
            # Update watchlist with last label for hysteresis
            if isinstance(watchlist.get(ticker), dict):
                watchlist[ticker]['_last_label'] = signal
            
            # Support calculation (use improved stop from signal)
            support = sig["risk"]["stop"]
            
            # Price change
            price_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
            
            # Analyst data
            analyst_rating = info.get('recommendationKey', 'N/A').upper()
            target = info.get('targetMeanPrice', 0)
            upside = ((target - latest['Close']) / latest['Close']) * 100 if target else 0
            
            watchlist_data.append({
                'Ticker': ticker,
                'Price': f"${latest['Close']:.2f}",
                'Change': f"{price_change:+.2f}%",
                'Signal': signal,
                'Score': f"{score}/100",
                'RSI': f"{latest['RSI']:.1f}",
                'Support': f"${support:.2f}",
                'Target (12mo)': f"${target:.2f}" if target else 'N/A',
                'Upside': f"{upside:+.1f}%" if target else 'N/A',
                'Analyst': analyst_rating,
                'Added': watchlist[ticker].get('added', 'N/A'),
                '_price_num': latest['Close'],
                '_change_num': price_change,
                '_signal': signal,
                '_score': score,
                '_reasons': reasons
            })
        except Exception as e:
            st.warning(f"Could not load {ticker}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(watchlist))
    
    progress_bar.empty()
    status_text.empty()
    
    if not watchlist_data:
        st.error("Could not load any watchlist stocks")
        return
    
    # Display as dataframe
    df_watchlist = pd.DataFrame(watchlist_data)
    
    # Color code by signal
    def color_signal(val):
        if val == 'BUY':
            return 'background-color: #90EE90'
        elif val == 'SELL':
            return 'background-color: #FFB6C1'
        else:
            return 'background-color: #FFE4B5'
    
    def color_change(val):
        try:
            num = float(val.replace('%', '').replace('+', ''))
            if num > 0:
                return 'color: green; font-weight: bold'
            elif num < 0:
                return 'color: red; font-weight: bold'
        except:
            pass
        return ''
    
    st.dataframe(
        df_watchlist[['Ticker', 'Price', 'Change', 'Signal', 'Score', 'RSI', 'Support', 'Target (12mo)', 'Upside', 'Analyst', 'Added']].style
            .applymap(color_signal, subset=['Signal'])
            .applymap(color_change, subset=['Change']),
        width='stretch',
        height=600
    )
    
    # Summary stats
    st.markdown("---")
    st.subheader("Watchlist Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        buy_count = len([d for d in watchlist_data if d['_signal'] == 'BUY'])
        st.metric("BUY Signals", buy_count)
    
    with col2:
        hold_count = len([d for d in watchlist_data if d['_signal'] == 'HOLD'])
        st.metric("HOLD Signals", hold_count)
    
    with col3:
        sell_count = len([d for d in watchlist_data if d['_signal'] == 'SELL'])
        st.metric("SELL Signals", sell_count)
    
    with col4:
        avg_change = sum([d['_change_num'] for d in watchlist_data]) / len(watchlist_data)
        st.metric("Avg Change", f"{avg_change:+.2f}%")
    
    # Action items
    st.markdown("---")
    st.subheader("Action Items")
    
    buy_signals = [d for d in watchlist_data if d['_signal'] == 'BUY']
    if buy_signals:
        st.success(f"**{len(buy_signals)} stocks with BUY signals:**")
        for stock in buy_signals:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"• **{stock['Ticker']}**: {stock['Price']} -> Target {stock['Target (12mo)']} ({stock['Upside']} upside)")
            with col_b:
                if st.button(f"Why?", key=f"why_buy_{stock['Ticker']}"):
                    with st.expander(f"{stock['Ticker']} Signal Details", expanded=True):
                        st.write(f"**Score: {stock['Score']}**")
                        for reason in stock.get('_reasons', [])[:8]:
                            st.write(reason)
    
    sell_signals = [d for d in watchlist_data if d['_signal'] == 'SELL']
    if sell_signals:
        st.error(f"**{len(sell_signals)} stocks with SELL signals:**")
        for stock in sell_signals:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"• **{stock['Ticker']}**: {stock['Price']} ({stock['Change']})")
            with col_b:
                if st.button(f"Why?", key=f"why_sell_{stock['Ticker']}"):
                    with st.expander(f"{stock['Ticker']} Signal Details", expanded=True):
                        st.write(f"**Score: {stock['Score']}**")
                        for reason in stock.get('_reasons', [])[:8]:
                            st.write(reason)
    
    # Signal reasoning details
    st.markdown("---")
    with st.expander("View All Signal Details", expanded=False):
        for stock in watchlist_data:
            st.markdown(f"### {stock['Ticker']} - {stock['Signal']} ({stock['Score']})")
            st.caption(f"Price: {stock['Price']} | RSI: {stock['RSI']}")
            for reason in stock.get('_reasons', []):
                st.write(f"  {reason}")
            st.markdown("---")
    
    # Back button at bottom
    st.markdown("---")
    if st.button("← Back to Dashboard", type="primary", key="back_bottom_wl"):
        st.session_state['show_watchlist'] = False
        st.rerun()

def show_portfolio_view():
    """Display portfolio tracker view"""
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        if st.button("← Back to Dashboard", type="primary", use_container_width=True):
            st.session_state['show_portfolio'] = False
            st.rerun()
    with col2:
        st.markdown("## Portfolio Tracker")
    with col3:
        if st.button("+ Add Position", key="add_pos_top"):
            st.session_state['show_add_position'] = True
    
    st.markdown("---")
    
    # Show add position form if requested
    if st.session_state.get('show_add_position', False):
        with st.expander("Add New Position", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                ticker = st.text_input("Ticker", key="new_ticker").upper()
                position_type = st.selectbox("Type", ["shares", "options"], key="new_type")
                quantity = st.number_input("Quantity", min_value=1, value=100, key="new_qty")
            
            with col2:
                entry_price = st.number_input("Entry Price ($)", min_value=0.01, value=100.0, step=0.01, key="new_price")
                entry_date = st.date_input("Entry Date", value=datetime.now(), key="new_date")
            
            with col3:
                if position_type == "options":
                    option_type = st.selectbox("Option Type", ["call", "put"], key="new_opt_type")
                    strike = st.number_input("Strike Price ($)", min_value=0.01, value=100.0, step=0.01, key="new_strike")
                    expiration = st.date_input("Expiration", value=datetime.now() + timedelta(days=30), key="new_exp")
                else:
                    option_type = None
                    strike = None
                    expiration = None
            
            col_submit, col_cancel = st.columns([1, 1])
            with col_submit:
                if st.button("Add Position", type="primary", use_container_width=True):
                    if ticker:
                        pt.add_position(
                            ticker=ticker,
                            position_type=position_type,
                            quantity=quantity,
                            entry_price=entry_price,
                            entry_date=str(entry_date),
                            strike=strike,
                            expiration=str(expiration) if expiration else None,
                            option_type=option_type
                        )
                        st.success(f"Added {ticker} to portfolio!")
                        st.session_state['show_add_position'] = False
                        st.rerun()
                    else:
                        st.error("Please enter a ticker")
            with col_cancel:
                if st.button("Cancel", use_container_width=True):
                    st.session_state['show_add_position'] = False
                    st.rerun()
    
    # Get portfolio summary
    with st.spinner("Loading portfolio..."):
        portfolio_data = pt.get_portfolio_summary()
    
    # Summary metrics
    summary = portfolio_data['summary']
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Value", f"${summary['total_value']:,.2f}")
    with col2:
        pnl_color = "normal" if summary['total_pnl'] >= 0 else "inverse"
        st.metric("Total P&L", f"${summary['total_pnl']:,.2f}", 
                 f"{summary['total_pnl_pct']:+.1f}%", delta_color=pnl_color)
    with col3:
        st.metric("Positions", summary['num_positions'])
    with col4:
        critical_count = summary['critical_alerts']
        warning_count = summary['warning_alerts']
        if critical_count > 0:
            st.metric("Alerts", f"{critical_count} Critical", delta_color="inverse")
        elif warning_count > 0:
            st.metric("Alerts", f"{warning_count} Warning", delta_color="off")
        else:
            st.metric("Alerts", "All Good", delta_color="normal")
    
    st.markdown("---")
    
    # Alert Feed
    alerts = portfolio_data['alerts']
    if alerts:
        st.subheader(f"Alert Feed ({len(alerts)} alerts)")
        
        # Group by severity
        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        warning_alerts = [a for a in alerts if a['severity'] == 'warning']
        info_alerts = [a for a in alerts if a['severity'] in ['info', 'positive']]
        
        if critical_alerts:
            with st.expander(f"[CRITICAL] Alerts ({len(critical_alerts)})", expanded=True):
                for alert in critical_alerts:
                    st.error(f"**{alert['message']}**")
                    st.caption(f"• {alert['detail']}")
                    st.caption(f"Action: {alert['action']}")
                    st.markdown("---")
        
        if warning_alerts:
            with st.expander(f"[WARNING] Alerts ({len(warning_alerts)})", expanded=True):
                for alert in warning_alerts:
                    st.warning(f"**{alert['message']}**")
                    st.caption(f"• {alert['detail']}")
                    st.caption(f"Action: {alert['action']}")
                    st.markdown("---")
        
        if info_alerts:
            with st.expander(f"[INFO] Opportunities ({len(info_alerts)})", expanded=False):
                for alert in info_alerts:
                    if alert['severity'] == 'positive':
                        st.success(f"**{alert['message']}**")
                    else:
                        st.info(f"**{alert['message']}**")
                    st.caption(f"• {alert['detail']}")
                    st.caption(f"Action: {alert['action']}")
                    st.markdown("---")
    
    st.markdown("---")
    
    # Positions Table
    st.subheader("Your Positions")
    
    if not portfolio_data['positions']:
        st.info("No positions yet. Click '+ Add Position' to start tracking!")
        return
    
    for pos_data in portfolio_data['positions']:
        position = pos_data['position']
        value = pos_data['value_data']
        
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 2, 2, 1])
            
            with col1:
                st.markdown(f"### {position['ticker']}")
                if position['type'] == 'options':
                    st.caption(f"{position['option_type'].upper()} ${position['strike']} exp {position['expiration']}")
                else:
                    st.caption(f"{position['quantity']} shares")
            
            with col2:
                st.metric("Entry", f"${position['entry_price']:.2f}")
                st.caption(f"Date: {position['entry_date']}")
            
            with col3:
                st.metric("Current", f"${pos_data['current_price']:.2f}")
                st.caption(f"RSI: {pos_data['rsi']:.1f}")
            
            with col4:
                pnl_color = "normal" if value['pnl'] >= 0 else "inverse"
                st.metric("P&L", f"${value['pnl']:,.2f}", 
                         f"{value['pnl_pct']:+.1f}%", delta_color=pnl_color)
            
            with col5:
                signal = pos_data['signal']
                if signal in ["STRONG BUY", "BUY"]:
                    st.success(f"**{signal}**")
                elif signal in ["SELL"]:
                    st.error(f"**{signal}**")
                else:
                    st.warning(f"**{signal}**")
                st.caption(f"Score: {pos_data['score']}/5")
            
            with col6:
                if st.button("Delete", key=f"del_pos_{position['id']}"):
                    pt.remove_position(position['id'])
                    st.rerun()
                if st.button("View", key=f"view_pos_{position['id']}"):
                    st.session_state['current_ticker'] = position['ticker']
                    st.session_state['show_portfolio'] = False
                    st.rerun()
            
            # Show position-specific alerts
            pos_alerts = pos_data['alerts']
            if pos_alerts:
                alert_count = len(pos_alerts)
                with st.expander(f"{alert_count} alert{'s' if alert_count > 1 else ''}", expanded=False):
                    for alert in pos_alerts:
                        st.caption(f"• {alert['message']}")
            
            st.markdown("---")
    
    # Back button at bottom
    st.markdown("---")
    if st.button("← Back to Dashboard", type="primary", key="back_bottom_portfolio"):
        st.session_state['show_portfolio'] = False
        st.rerun()

def show_paper_trading_view():
    """Display paper trading view"""
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        if st.button("← Back to Dashboard", type="primary", use_container_width=True):
            st.session_state['show_paper_trading'] = False
            st.rerun()
    with col2:
        st.markdown("## Paper Trading")
    with col3:
        if st.button("Reset Portfolio", key="reset_paper"):
            st.session_state['show_reset_confirm'] = True
    
    st.markdown("---")
    
    # Initialize engine
    paper_engine = PaperTradingEngine()
    
    # Portfolio Summary
    portfolio = paper_engine.get_portfolio_value()
    performance = paper_engine.get_performance_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Value", f"${portfolio['total_value']:,.2f}")
    with col2:
        pnl_color = "normal" if portfolio['total_pnl'] >= 0 else "inverse"
        st.metric("Total P&L", f"${portfolio['total_pnl']:,.2f}", 
                 f"{portfolio['total_pnl_pct']:+.2f}%", delta_color=pnl_color)
    with col3:
        st.metric("Cash", f"${portfolio['cash']:,.2f}")
    with col4:
        st.metric("Positions", portfolio['num_positions'])
    
    # Reset confirmation
    if st.session_state.get('show_reset_confirm', False):
        st.warning("Are you sure you want to reset your paper trading portfolio? This cannot be undone.")
        col_reset1, col_reset2 = st.columns(2)
        with col_reset1:
            if st.button("Yes, Reset Portfolio", type="primary"):
                paper_engine.reset_portfolio(100000)
                st.session_state['show_reset_confirm'] = False
                st.success("Portfolio reset to $100,000")
                st.rerun()
        with col_reset2:
            if st.button("Cancel"):
                st.session_state['show_reset_confirm'] = False
                st.rerun()
    
    st.markdown("---")
    
    # Trading Interface
    tab1, tab2, tab3 = st.tabs(["Trade", "Positions", "History"])
    
    # TAB 1: Trading
    with tab1:
        st.subheader("Execute Trade")
        
        col_trade1, col_trade2 = st.columns(2)
        
        with col_trade1:
            st.markdown("### Buy Stock")
            buy_ticker = st.text_input("Ticker", key="buy_ticker").upper()
            buy_shares = st.number_input("Shares", min_value=1, value=100, key="buy_shares")
            
            if buy_ticker:
                current_price = paper_engine.get_current_price(buy_ticker)
                if current_price:
                    total_cost = buy_shares * current_price
                    st.write(f"**Current Price:** ${current_price:.2f}")
                    st.write(f"**Total Cost:** ${total_cost:,.2f}")
                    
                    if st.button("Buy", type="primary", key="execute_buy"):
                        result = paper_engine.buy(buy_ticker, buy_shares)
                        if result['success']:
                            st.success(result['message'])
                            st.rerun()
                        else:
                            st.error(result['message'])
                else:
                    st.warning("Could not fetch price for this ticker")
        
        with col_trade2:
            st.markdown("### Sell Stock")
            
            # Get list of owned stocks
            positions = portfolio['positions']
            if positions:
                sell_ticker = st.selectbox("Select Position", list(positions.keys()), key="sell_ticker")
                
                if sell_ticker:
                    pos = positions[sell_ticker]
                    max_shares = int(pos['shares'])
                    sell_shares = st.number_input("Shares", min_value=1, max_value=max_shares, value=min(100, max_shares), key="sell_shares")
                    
                    current_price = pos['current_price']
                    total_proceeds = sell_shares * current_price
                    potential_pnl = (current_price - pos['avg_price']) * sell_shares
                    potential_pnl_pct = (potential_pnl / (pos['avg_price'] * sell_shares)) * 100
                    
                    st.write(f"**Current Price:** ${current_price:.2f}")
                    st.write(f"**Avg Cost:** ${pos['avg_price']:.2f}")
                    st.write(f"**Proceeds:** ${total_proceeds:,.2f}")
                    st.write(f"**Est. P&L:** ${potential_pnl:,.2f} ({potential_pnl_pct:+.2f}%)")
                    
                    if st.button("Sell", type="primary", key="execute_sell"):
                        result = paper_engine.sell(sell_ticker, sell_shares)
                        if result['success']:
                            st.success(result['message'])
                            st.rerun()
                        else:
                            st.error(result['message'])
            else:
                st.info("No positions to sell. Buy some stocks first!")
    
    # TAB 2: Positions
    with tab2:
        st.subheader("Current Positions")
        
        if portfolio['positions']:
            positions_data = []
            for ticker, pos in portfolio['positions'].items():
                positions_data.append({
                    'Ticker': ticker,
                    'Shares': pos['shares'],
                    'Avg Price': f"${pos['avg_price']:.2f}",
                    'Current Price': f"${pos['current_price']:.2f}",
                    'Market Value': f"${pos['market_value']:,.2f}",
                    'Cost Basis': f"${pos['cost_basis']:,.2f}",
                    'P&L': f"${pos['pnl']:,.2f}",
                    'P&L %': f"{pos['pnl_pct']:+.2f}%",
                    '_pnl_num': pos['pnl']
                })
            
            df_positions = pd.DataFrame(positions_data)
            
            # Color code P&L
            def color_pnl(val):
                try:
                    if '+' in val:
                        return 'color: green; font-weight: bold'
                    elif '-' in val:
                        return 'color: red; font-weight: bold'
                except:
                    pass
                return ''
            
            st.dataframe(
                df_positions[['Ticker', 'Shares', 'Avg Price', 'Current Price', 'Market Value', 'P&L', 'P&L %']]
                    .style.applymap(color_pnl, subset=['P&L', 'P&L %']),
                use_container_width=True
            )
            
            # Summary stats
            st.markdown("---")
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            
            with col_sum1:
                total_market_value = sum(p['market_value'] for p in portfolio['positions'].values())
                st.metric("Total Market Value", f"${total_market_value:,.2f}")
            
            with col_sum2:
                total_cost = sum(p['cost_basis'] for p in portfolio['positions'].values())
                st.metric("Total Cost Basis", f"${total_cost:,.2f}")
            
            with col_sum3:
                unrealized_pnl = total_market_value - total_cost
                unrealized_pnl_pct = (unrealized_pnl / total_cost) * 100 if total_cost > 0 else 0
                pnl_color = "normal" if unrealized_pnl >= 0 else "inverse"
                st.metric("Unrealized P&L", f"${unrealized_pnl:,.2f}", 
                         f"{unrealized_pnl_pct:+.2f}%", delta_color=pnl_color)
        else:
            st.info("No open positions. Start trading to see your positions here!")
    
    # TAB 3: History
    with tab3:
        st.subheader("Trade History")
        
        trades = paper_engine.get_trade_history(limit=50)
        
        if trades:
            # Performance summary
            st.markdown("### Performance Summary")
            
            col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
            
            with col_perf1:
                st.metric("Total Trades", performance['total_trades'])
            with col_perf2:
                st.metric("Win Rate", f"{performance['win_rate']:.1f}%")
            with col_perf3:
                if 'total_realized_pnl' in performance:
                    pnl_color = "normal" if performance['total_realized_pnl'] >= 0 else "inverse"
                    st.metric("Realized P&L", f"${performance['total_realized_pnl']:,.2f}", delta_color=pnl_color)
            with col_perf4:
                st.metric("Avg Win", f"${performance['avg_win']:,.2f}")
            
            # Trade log
            st.markdown("---")
            st.markdown("### Recent Trades")
            
            for trade in trades[:20]:
                with st.container():
                    col_log1, col_log2, col_log3 = st.columns([2, 2, 2])
                    
                    timestamp = datetime.fromisoformat(trade['timestamp'])
                    
                    with col_log1:
                        if trade['type'] == 'BUY':
                            st.success(f"**{trade['type']} {trade['ticker']}**")
                        else:
                            st.error(f"**{trade['type']} {trade['ticker']}**")
                        st.caption(timestamp.strftime('%Y-%m-%d %H:%M'))
                    
                    with col_log2:
                        st.write(f"{trade['shares']} shares @ ${trade['price']:.2f}")
                        st.caption(f"Total: ${trade['total']:,.2f}")
                    
                    with col_log3:
                        if trade['type'] == 'SELL' and 'pnl' in trade:
                            pnl_color = "green" if trade['pnl'] > 0 else "red"
                            st.markdown(f"<span style='color:{pnl_color};font-weight:bold;'>P&L: ${trade['pnl']:,.2f} ({trade['pnl_pct']:+.2f}%)</span>", unsafe_allow_html=True)
                    
                    st.markdown("---")
        else:
            st.info("No trades yet. Start trading to see your history here!")
    
    # Back button at bottom
    st.markdown("---")
    if st.button("← Back to Dashboard", type="primary", key="back_bottom_paper"):
        st.session_state['show_paper_trading'] = False
        st.rerun()

def show_backtest_view():
    """Display backtest view"""
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Dashboard", type="primary", use_container_width=True):
            st.session_state['show_backtest'] = False
            st.rerun()
    with col2:
        st.markdown("## Strategy Backtester")
    
    st.markdown("---")
    
    # Tabs for different backtest modes
    tab1, tab2, tab3 = st.tabs(["Single Backtest", "Parameter Optimization", "Walk-Forward Test"])
    
    # === TAB 1: Single Backtest ===
    with tab1:
        st.subheader("Backtest Trading Strategy")
        
        # Configuration
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bt_ticker = st.text_input("Ticker", value="NTLA", key="bt_ticker").upper()
        with col2:
            bt_period = st.selectbox("Period", ['1y', '2y', '3y', '5y'], index=1, key="bt_period")
        with col3:
            bt_capital = st.number_input("Initial Capital ($)", value=10000, step=1000, key="bt_capital")
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            run_backtest = st.button("Run Backtest", type="primary", use_container_width=True)
        
        # Strategy parameters
        with st.expander("Strategy Parameters", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_low = st.slider("RSI Low", 20, 50, 40, key="rsi_low")
                rsi_high = st.slider("RSI High", 50, 80, 60, key="rsi_high")
            with col2:
                adx_threshold = st.slider("ADX Threshold", 20, 40, 25, key="adx_threshold")
            with col3:
                stop_loss = st.slider("Stop Loss %", 3, 15, 8, key="stop_loss")
                take_profit = st.slider("Take Profit %", 10, 50, 20, key="take_profit")
        
        if run_backtest or st.session_state.get('backtest_results'):
            try:
                with st.spinner(f"Running backtest on {bt_ticker}..."):
                    # Initialize engine
                    engine = BacktestEngine(bt_ticker, bt_capital)
                    engine.set_parameters(
                        rsi_low=rsi_low,
                        rsi_high=rsi_high,
                        adx_threshold=adx_threshold,
                        stop_loss_pct=stop_loss,
                        take_profit_pct=take_profit
                    )
                    
                    # Download and run
                    engine.download_data(period=bt_period)
                    results = engine.run_backtest()
                    
                    # Store in session state
                    if run_backtest:
                        st.session_state['backtest_results'] = results
                    else:
                        results = st.session_state['backtest_results']
                
                # Display results
                st.markdown("---")
                st.subheader("Backtest Results")
                
                metrics = results['metrics']
                df_trades = results['trades']
                df_portfolio = results['portfolio_value']
                
                # Key Metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    return_color = "normal" if metrics['total_return'] >= 0 else "inverse"
                    st.metric("Total Return", f"{metrics['total_return']:.2f}%", delta_color=return_color)
                
                with col2:
                    st.metric("Buy & Hold", f"{metrics['buy_hold_return']:.2f}%")
                
                with col3:
                    outperf_color = "normal" if metrics['outperformance'] >= 0 else "inverse"
                    st.metric("Outperformance", f"{metrics['outperformance']:+.2f}%", delta_color=outperf_color)
                
                with col4:
                    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                
                with col5:
                    st.metric("Num Trades", f"{metrics['num_trades']}")
                
                # Advanced Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    sharpe_color = "normal" if metrics['sharpe_ratio'] > 1 else "off"
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}", help="Risk-adjusted return (>1 is good)", delta_color=sharpe_color)
                
                with col2:
                    st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%", delta_color="inverse")
                
                with col3:
                    st.metric("Avg Win", f"{metrics['avg_win']:.2f}%", delta_color="normal")
                
                with col4:
                    st.metric("Avg Loss", f"{metrics['avg_loss']:.2f}%", delta_color="inverse")
                
                # Equity Curve Chart
                st.markdown("---")
                st.subheader("Equity Curve")
                
                fig_equity = go.Figure()
                
                # Portfolio value
                fig_equity.add_trace(go.Scatter(
                    x=df_portfolio['Date'],
                    y=df_portfolio['Value'],
                    name='Strategy',
                    line=dict(color='blue', width=2)
                ))
                
                # Buy & Hold comparison
                df_backtest = results['df']
                buy_hold_values = (df_backtest['Close'] / df_backtest['Close'].iloc[0]) * bt_capital
                fig_equity.add_trace(go.Scatter(
                    x=df_backtest.index,
                    y=buy_hold_values,
                    name='Buy & Hold',
                    line=dict(color='gray', width=2, dash='dash')
                ))
                
                fig_equity.update_layout(
                    title=f"{bt_ticker} - Strategy vs Buy & Hold",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig_equity, use_container_width=True)
                
                # Price Chart with Trade Markers
                st.markdown("---")
                st.subheader("Price Chart with Trades")
                
                fig_trades = go.Figure()
                
                # Price line
                fig_trades.add_trace(go.Scatter(
                    x=df_backtest.index,
                    y=df_backtest['Close'],
                    name='Price',
                    line=dict(color='lightblue', width=1)
                ))
                
                # Buy signals
                buy_trades = df_trades[df_trades['Type'] == 'BUY']
                if len(buy_trades) > 0:
                    fig_trades.add_trace(go.Scatter(
                        x=buy_trades['Date'],
                        y=buy_trades['Price'],
                        mode='markers',
                        name='Buy',
                        marker=dict(symbol='triangle-up', size=12, color='green')
                    ))
                
                # Sell signals
                sell_trades = df_trades[df_trades['Type'] == 'SELL']
                if len(sell_trades) > 0:
                    # Color by profit/loss
                    colors = ['green' if pnl > 0 else 'red' for pnl in sell_trades['PnL%']]
                    fig_trades.add_trace(go.Scatter(
                        x=sell_trades['Date'],
                        y=sell_trades['Price'],
                        mode='markers',
                        name='Sell',
                        marker=dict(symbol='triangle-down', size=12, color=colors),
                        text=[f"{pnl:+.1f}% ({reason})" for pnl, reason in zip(sell_trades['PnL%'], sell_trades['Reason'])],
                        hovertemplate='%{text}<extra></extra>'
                    ))
                
                fig_trades.update_layout(
                    title=f"{bt_ticker} - Trade Entry & Exit Points",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='closest',
                    height=500
                )
                
                st.plotly_chart(fig_trades, use_container_width=True)
                
                # Trade Log
                st.markdown("---")
                st.subheader("Trade Log")
                
                if len(df_trades) > 0:
                    # Format for display
                    display_trades = df_trades.copy()
                    display_trades['Date'] = display_trades['Date'].dt.strftime('%Y-%m-%d')
                    display_trades['Price'] = display_trades['Price'].apply(lambda x: f"${x:.2f}")
                    display_trades['Shares'] = display_trades['Shares'].apply(lambda x: f"{x:.2f}")
                    display_trades['PnL%'] = display_trades['PnL%'].apply(lambda x: f"{x:+.2f}%")
                    
                    st.dataframe(
                        display_trades[['Date', 'Type', 'Price', 'Shares', 'PnL%', 'Reason']],
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.info("No trades generated. Try adjusting the strategy parameters.")
                
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")
    
    # === TAB 2: Parameter Optimization ===
    with tab2:
        st.subheader("Optimize Strategy Parameters")
        st.caption("Test multiple parameter combinations to find the best settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            opt_ticker = st.text_input("Ticker", value="NTLA", key="opt_ticker").upper()
        with col2:
            opt_period = st.selectbox("Period", ['1y', '2y', '3y'], index=1, key="opt_period")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_optimization = st.button("Run Optimization", type="primary", use_container_width=True)
        
        # Parameter ranges
        with st.expander("Parameter Ranges to Test", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                stop_loss_range = st.multiselect(
                    "Stop Loss % to test",
                    [3, 5, 8, 10, 12, 15],
                    default=[5, 8, 10]
                )
                rsi_low_range = st.multiselect(
                    "RSI Low to test",
                    [30, 35, 40, 45],
                    default=[35, 40]
                )
            with col2:
                take_profit_range = st.multiselect(
                    "Take Profit % to test",
                    [10, 15, 20, 25, 30],
                    default=[15, 20, 25]
                )
                rsi_high_range = st.multiselect(
                    "RSI High to test",
                    [55, 60, 65, 70],
                    default=[60, 65]
                )
        
        if run_optimization:
            try:
                with st.spinner(f"Optimizing parameters for {opt_ticker}... This may take a minute"):
                    engine = BacktestEngine(opt_ticker, 10000)
                    engine.download_data(period=opt_period)
                    
                    param_grid = {
                        'stop_loss_pct': stop_loss_range,
                        'take_profit_pct': take_profit_range,
                        'rsi_low': rsi_low_range,
                        'rsi_high': rsi_high_range
                    }
                    
                    opt_results = engine.optimize_parameters(param_grid)
                    
                    st.success(f"Tested {len(opt_results)} parameter combinations!")
                    
                    # Sort by total return
                    opt_results = opt_results.sort_values('total_return', ascending=False)
                    
                    # Top 10 results
                    st.subheader("Top 10 Parameter Combinations")
                    
                    display_results = opt_results.head(10).copy()
                    display_results = display_results[[
                        'stop_loss_pct', 'take_profit_pct', 'rsi_low', 'rsi_high',
                        'total_return', 'win_rate', 'num_trades', 'sharpe_ratio', 'max_drawdown'
                    ]]
                    
                    # Format columns
                    display_results['total_return'] = display_results['total_return'].apply(lambda x: f"{x:.2f}%")
                    display_results['win_rate'] = display_results['win_rate'].apply(lambda x: f"{x:.1f}%")
                    display_results['sharpe_ratio'] = display_results['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
                    display_results['max_drawdown'] = display_results['max_drawdown'].apply(lambda x: f"{x:.2f}%")
                    
                    st.dataframe(display_results, use_container_width=True)
                    
                    # Best parameters
                    best = opt_results.iloc[0]
                    st.success(f"""
                        **Best Parameters Found:**
                        - Stop Loss: {best['stop_loss_pct']}%
                        - Take Profit: {best['take_profit_pct']}%
                        - RSI Range: {best['rsi_low']}-{best['rsi_high']}
                        
                        **Performance:**
                        - Return: {best['total_return']:.2f}%
                        - Win Rate: {best['win_rate']:.1f}%
                        - Sharpe Ratio: {best['sharpe_ratio']:.2f}
                    """)
                    
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
    
    # === TAB 3: Walk-Forward Test ===
    with tab3:
        st.subheader("Walk-Forward Testing")
        st.caption("Train on first 70% of data, test on last 30% - more realistic performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            wf_ticker = st.text_input("Ticker", value="NTLA", key="wf_ticker").upper()
        with col2:
            train_ratio = st.slider("Train/Test Split", 0.5, 0.8, 0.7, 0.05, key="train_ratio",
                                   help="0.7 = train on first 70%, test on last 30%")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_walkforward = st.button("Run Walk-Forward", type="primary", use_container_width=True)
        
        if run_walkforward:
            try:
                with st.spinner(f"Running walk-forward test on {wf_ticker}..."):
                    engine = BacktestEngine(wf_ticker, 10000)
                    wf_results = engine.walk_forward_test(train_ratio=train_ratio)
                    
                    train_metrics = wf_results['train']['metrics']
                    test_metrics = wf_results['test']['metrics']
                    
                    st.success("Walk-forward test complete!")
                    
                    # Compare train vs test
                    st.subheader("Train vs Test Performance")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Training Period")
                        st.metric("Return", f"{train_metrics['total_return']:.2f}%")
                        st.metric("Win Rate", f"{train_metrics['win_rate']:.1f}%")
                        st.metric("Sharpe Ratio", f"{train_metrics['sharpe_ratio']:.2f}")
                        st.metric("Num Trades", f"{train_metrics['num_trades']}")
                    
                    with col2:
                        st.markdown("### Test Period (Out-of-Sample)")
                        return_color = "normal" if test_metrics['total_return'] >= 0 else "inverse"
                        st.metric("Return", f"{test_metrics['total_return']:.2f}%", delta_color=return_color)
                        st.metric("Win Rate", f"{test_metrics['win_rate']:.1f}%")
                        st.metric("Sharpe Ratio", f"{test_metrics['sharpe_ratio']:.2f}")
                        st.metric("Num Trades", f"{test_metrics['num_trades']}")
                    
                    # Degradation analysis
                    degradation = test_metrics['total_return'] - train_metrics['total_return']
                    
                    if degradation < -20:
                        st.error(f"""
                            **Overfitting Detected!**
                            
                            The strategy performed {abs(degradation):.1f}% worse on unseen data.
                            This suggests the parameters are overfit to historical data.
                            
                            Recommendation: Use more conservative parameters or longer training period.
                        """)
                    elif degradation < -10:
                        st.warning(f"""
                            **Moderate Degradation**
                            
                            Performance dropped {abs(degradation):.1f}% on unseen data.
                            This is somewhat expected but monitor closely.
                        """)
                    else:
                        st.success(f"""
                            **Robust Strategy!**
                            
                            Test performance is similar to training ({degradation:+.1f}% difference).
                            This suggests the strategy generalizes well.
                        """)
                    
            except Exception as e:
                st.error(f"Error during walk-forward test: {str(e)}")
    
    # Back button at bottom
    st.markdown("---")
    if st.button("← Back to Dashboard", type="primary", key="back_bottom_backtest"):
        st.session_state['show_backtest'] = False
        st.rerun()

def show_comparison_view():
    """Show stock comparison view"""
    # Back button at the top
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("← Back to Dashboard", type="primary", use_container_width=True):
            st.session_state['show_comparison'] = False
            st.rerun()
    with col2:
        st.title("Stock Comparison")
    
    compare_tickers = st.session_state.get('compare_tickers', [])
    
    if not compare_tickers or len(compare_tickers) < 2:
        st.warning("Please enter at least 2 tickers to compare")
        return
    
    st.success(f"Comparing: {', '.join(compare_tickers)}")
    st.caption("Click '← Back to Dashboard' to return to main analysis")
    
    # Load data for all tickers
    comparison_data = []
    cols = st.columns(len(compare_tickers))
    
    for i, ticker in enumerate(compare_tickers):
        with cols[i]:
            try:
                df, info = load_stock_data(ticker, period='3mo')
                
                if df.empty:
                    st.error(f"No data for {ticker}")
                    continue
                    
                df = calculate_all_indicators(df)
                latest = df.iloc[-1]
                
                # Get signal params from session state
                params = st.session_state.get('signal_params', SignalParams())
                
                # Compute signal using new system
                sig = compute_signal(df, info, params, last_label=None)
                signal = sig["label"]
                score = sig["score"]
                reasons = sig["reasons"]
                
                analyst_rating = info.get('recommendationKey', 'N/A').upper()
                
                # Display card
                st.markdown(f"### {ticker}")
                st.metric("Price", f"${latest['Close']:.2f}")
                st.metric("Score", f"{score}/100")
                
                if signal == "BUY":
                    st.success(signal)
                elif signal == "SELL":
                    st.error(signal)
                else:
                    st.warning(signal)
                
                st.caption(f"Analyst: {analyst_rating}")
                st.caption(f"RSI: {latest['RSI']:.1f}")
                st.caption(f"ADX: {latest['ADX']:.1f}")
                
                upside = 0
                if info.get('targetMeanPrice'):
                    upside = ((info['targetMeanPrice'] - latest['Close']) / latest['Close']) * 100
                    st.metric(
                        "Upside", 
                        f"{upside:+.1f}%",
                        help="To 12-month analyst target"
                    )
                    st.caption(f"Target (12-mo): ${info['targetMeanPrice']:.2f}")
                
                # Show signal reasoning
                with st.expander("Why this signal?", expanded=False):
                    for reason in reasons[:6]:
                        st.write(f"• {reason}")
                    st.caption(f"Stop: ${sig['risk']['stop']:.2f} | Target: ${sig['risk']['tgt1']:.2f}")
                
                if st.button(f"Analyze {ticker}", key=f"analyze_{ticker}"):
                    st.session_state['current_ticker'] = ticker
                    st.session_state['show_comparison'] = False
                    st.rerun()
                
                comparison_data.append({
                    'ticker': ticker,
                    'price': latest['Close'],
                    'signal': signal,
                    'score': score,
                    'rsi': latest['RSI'],
                    'upside': upside
                })
                
            except Exception as e:
                st.error(f"Error loading {ticker}")
                st.caption(str(e))
    
    # Summary table
    if comparison_data:
        st.markdown("---")
        st.subheader("Comparison Summary")
        df_compare = pd.DataFrame(comparison_data)
        st.dataframe(df_compare.style.format({
            'price': '${:.2f}',
            'rsi': '{:.1f}',
            'upside': '{:+.1f}%'
        }), width='stretch')
        
        # Recommendation
        buy_signals = [d for d in comparison_data if d['signal'] == 'BUY']
        if buy_signals:
            best = max(buy_signals, key=lambda x: x['upside'])
            st.success(f"**Best Opportunity: {best['ticker']}** - {best['upside']:+.1f}% upside")
        else:
            st.info("No BUY signals found. Monitor for better entries.")
    
    # Back button at bottom too
    st.markdown("---")
    if st.button("← Back to Dashboard", type="primary", key="back_bottom"):
        st.session_state['show_comparison'] = False
        st.rerun()

def main():
    # Header
    st.markdown('<h1 class="main-header">Live Stock Trading Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Ticker selection
    default_tickers = ['NTLA', 'INSM', 'IONS', 'SANA', 'AAPL', 'TSLA', 'NVDA', 'MSFT']
    
    ticker_input = st.sidebar.text_input(
        "Enter Ticker Symbol", 
        value=st.session_state.get('current_ticker', 'NTLA'),
        help="Enter a stock ticker symbol (e.g., AAPL, TSLA)"
    )
    
    # Store current ticker
    st.session_state['current_ticker'] = ticker_input
    
    # Quick select buttons
    st.sidebar.write("Quick Select:")
    cols = st.sidebar.columns(2)
    for i, tick in enumerate(default_tickers):
        if cols[i % 2].button(tick, key=f"btn_{tick}"):
            ticker_input = tick
            st.session_state['current_ticker'] = tick
            st.rerun()
    
    # Watchlist Section
    st.sidebar.markdown("---")
    
    # Clickable header for watchlist
    col_wl1, col_wl2 = st.sidebar.columns([3, 1])
    with col_wl1:
        st.markdown("### Watchlist")
    with col_wl2:
        if st.button("View All", key="view_all_watchlist"):
            st.session_state['show_watchlist'] = True
            st.rerun()
    
    import json
    import os
    
    WATCHLIST_FILE = 'watchlist.json'
    
    # Initialize watchlist in session state if not exists
    if 'watchlist' not in st.session_state:
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, 'r') as f:
                st.session_state['watchlist'] = json.load(f)
        else:
            st.session_state['watchlist'] = {}
    
    watchlist = st.session_state['watchlist']
    
    # Add current stock to watchlist
    if st.sidebar.button("+ Add Current Stock", key="add_to_watchlist", use_container_width=True):
        if ticker_input and ticker_input not in watchlist:
            watchlist[ticker_input] = {
                'added': datetime.now().strftime('%Y-%m-%d'),
                'notes': ''
            }
            st.session_state['watchlist'] = watchlist
            with open(WATCHLIST_FILE, 'w') as f:
                json.dump(watchlist, f, indent=2)
            st.sidebar.success(f"Added {ticker_input}")
        elif ticker_input in watchlist:
            st.sidebar.info(f"{ticker_input} already saved")
    
    # Display watchlist with live data
    if watchlist:
        st.sidebar.caption(f"{len(watchlist)} stocks")
        
        for tick in list(watchlist.keys())[:8]:  # Show first 8
            # Get quick data
            quick_data = get_quick_stock_data(tick)
            
            if quick_data:
                # Create a styled button container
                col1, col2, col3 = st.sidebar.columns([2, 2, 1])
                
                with col1:
                    if st.button(f"**{tick}**", key=f"wl_{tick}", use_container_width=True):
                        ticker_input = tick
                        st.session_state['current_ticker'] = tick
                        st.rerun()
                
                with col2:
                    # Show price and change
                    change_color = "green" if quick_data['change'] > 0 else "red"
                    st.markdown(
                        f"<div style='font-size:11px;'>"
                        f"${quick_data['price']}<br>"
                        f"<span style='color:{change_color};font-weight:bold;'>{quick_data['change']:+.1f}%</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                with col3:
                    # Show signal as colored badge
                    if quick_data['signal'] == 'BUY':
                        st.markdown("<div style='background:green;color:white;padding:2px 4px;border-radius:3px;font-size:10px;text-align:center;'>BUY</div>", unsafe_allow_html=True)
                    elif quick_data['signal'] == 'SELL':
                        st.markdown("<div style='background:red;color:white;padding:2px 4px;border-radius:3px;font-size:10px;text-align:center;'>SELL</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='background:orange;color:white;padding:2px 4px;border-radius:3px;font-size:10px;text-align:center;'>HOLD</div>", unsafe_allow_html=True)
                    
                    # Delete button below
                    if st.button("X", key=f"del_{tick}"):
                        del watchlist[tick]
                        st.session_state['watchlist'] = watchlist
                        with open(WATCHLIST_FILE, 'w') as f:
                            json.dump(watchlist, f, indent=2)
                        st.rerun()
            else:
                # Fallback if data can't be loaded
                col1, col2 = st.sidebar.columns([4, 1])
                with col1:
                    if st.button(tick, key=f"wl_{tick}", use_container_width=True):
                        ticker_input = tick
                        st.session_state['current_ticker'] = tick
                        st.rerun()
                with col2:
                    if st.button("X", key=f"del_{tick}"):
                        del watchlist[tick]
                        st.session_state['watchlist'] = watchlist
                        with open(WATCHLIST_FILE, 'w') as f:
                            json.dump(watchlist, f, indent=2)
                        st.rerun()
        
        if len(watchlist) > 8:
            st.sidebar.caption(f"+ {len(watchlist) - 8} more (click 'View All')")
    else:
        st.sidebar.info("No stocks saved yet")
    
    # Portfolio Section
    st.sidebar.markdown("---")
    col_port1, col_port2 = st.sidebar.columns([3, 1])
    with col_port1:
        st.markdown("### Portfolio")
    with col_port2:
        if st.button("View", key="view_portfolio"):
            st.session_state['show_portfolio'] = True
            st.rerun()
    
    # Load portfolio for quick stats
    portfolio = pt.load_portfolio()
    if portfolio['positions']:
        num_positions = len(portfolio['positions'])
        st.sidebar.caption(f"{num_positions} position{'s' if num_positions != 1 else ''}")
        
        # Quick summary
        try:
            portfolio_data = pt.get_portfolio_summary()
            summary = portfolio_data['summary']
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                pnl_color = "green" if summary['total_pnl'] >= 0 else "red"
                st.sidebar.markdown(
                    f"<div style='font-size:12px;'>P&L: "
                    f"<span style='color:{pnl_color};font-weight:bold;'>{summary['total_pnl_pct']:+.1f}%</span></div>",
                    unsafe_allow_html=True
                )
            with col2:
                if summary['critical_alerts'] > 0:
                    st.sidebar.markdown(f"<div style='font-size:12px;color:red;font-weight:bold;'>{summary['critical_alerts']} alert{'s' if summary['critical_alerts'] != 1 else ''}</div>", unsafe_allow_html=True)
                elif summary['warning_alerts'] > 0:
                    st.sidebar.markdown(f"<div style='font-size:12px;color:orange;font-weight:bold;'>{summary['warning_alerts']} alert{'s' if summary['warning_alerts'] != 1 else ''}</div>", unsafe_allow_html=True)
                else:
                    st.sidebar.markdown("<div style='font-size:12px;color:green;'>All good</div>", unsafe_allow_html=True)
        except:
            pass
    else:
        st.sidebar.info("No positions tracked")
        if st.sidebar.button("+ Add First Position", use_container_width=True):
            st.session_state['show_portfolio'] = True
            st.session_state['show_add_position'] = True
            st.rerun()
    
    # Paper Trading Section
    st.sidebar.markdown("---")
    col_paper1, col_paper2 = st.sidebar.columns([3, 1])
    with col_paper1:
        st.markdown("### Paper Trading")
    with col_paper2:
        if st.button("Trade", key="view_paper_trading"):
            st.session_state['show_paper_trading'] = True
            st.rerun()
    
    st.sidebar.caption("Practice trading with virtual money")
    
    # Backtest Section
    st.sidebar.markdown("---")
    col_bt1, col_bt2 = st.sidebar.columns([3, 1])
    with col_bt1:
        st.markdown("### Backtest")
    with col_bt2:
        if st.button("Test", key="view_backtest"):
            st.session_state['show_backtest'] = True
            st.rerun()
    
    st.sidebar.caption("Test trading strategies on historical data")
    
    # Quick Compare Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Compare")
    
    compare_input = st.sidebar.text_input(
        "Compare tickers",
        value="NTLA,EDIT,CRSP",
        key="compare_input",
        help="Enter 2-5 tickers separated by commas"
    )
    
    if st.sidebar.button("Compare Now", key="compare_now", type="primary"):
        compare_tickers = [t.strip().upper() for t in compare_input.split(',') if t.strip()]
        if len(compare_tickers) >= 2:
            st.session_state['compare_tickers'] = compare_tickers
            st.session_state['show_comparison'] = True
            st.rerun()
        else:
            st.sidebar.error("Enter at least 2 tickers")
    
    # Preset comparisons
    st.sidebar.caption("**Quick Presets:**")
    preset_cols = st.sidebar.columns(3)
    
    presets = {
        'CRISPR': ['NTLA', 'EDIT', 'CRSP'],
        'Biotech': ['VRTX', 'REGN', 'ALNY'],
        'Tech': ['AAPL', 'MSFT', 'NVDA']
    }
    
    for i, (name, tickers) in enumerate(presets.items()):
        with preset_cols[i]:
            if st.button(name, key=f"preset_{name}", use_container_width=True):
                st.session_state['compare_tickers'] = tickers
                st.session_state['show_comparison'] = True
                st.rerun()
    
    # Time period
    period = st.sidebar.selectbox(
        "Time Period",
        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=2
    )
    
    # Refresh button
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", value=False)
    if auto_refresh:
        st.sidebar.info("Dashboard will refresh every 5 minutes")
    
    # Signal Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Signal Settings")
    
    with st.sidebar.expander("Customize Thresholds", expanded=False):
        rsi_lo, rsi_hi = st.slider(
            "RSI Band", 
            10, 90, (40, 60),
            help="RSI range considered healthy for entries"
        )
        adx_thr = st.slider(
            "ADX Threshold", 
            10, 50, 25,
            help="Minimum ADX for strong trend confirmation"
        )
        vol_mult = st.slider(
            "Volume Multiple", 
            0.5, 2.0, 1.0, 0.1,
            help="Required volume vs 20-day average"
        )
        use_weekly = st.checkbox(
            "Weekly Trend Confirmation", 
            value=True,
            help="Require weekly chart to confirm uptrend"
        )
        buy_threshold = st.slider(
            "Buy Score Threshold",
            50, 90, 70,
            help="Minimum score to generate BUY signal"
        )
        sell_threshold = st.slider(
            "Sell Score Threshold",
            10, 50, 30,
            help="Maximum score to generate SELL signal"
        )
    
    # Create signal params from user settings
    signal_params = SignalParams(
        rsi_band=(rsi_lo, rsi_hi),
        adx_thr=adx_thr,
        vol_mult=vol_mult,
        weekly_confirm=use_weekly,
        buy_score_thr=buy_threshold,
        sell_score_thr=sell_threshold,
    )
    
    # Store in session state for access in views
    st.session_state['signal_params'] = signal_params
    
    # Alerts Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Price Alerts")
    
    with st.sidebar.expander("Set Alert Conditions", expanded=False):
        enable_alerts = st.checkbox("Enable Alerts", value=False, key="enable_alerts")
        
        if enable_alerts:
            alert_type = st.selectbox(
                "Alert Type",
                ["Price Above", "Price Below", "RSI Overbought (>70)", "RSI Oversold (<30)", "Volume Spike (>2x avg)"],
                key="alert_type"
            )
            
            if alert_type in ["Price Above", "Price Below"]:
                alert_price = st.number_input(
                    "Alert Price ($)",
                    min_value=0.01,
                    value=100.0,
                    step=0.01,
                    key="alert_price"
                )
            
            # Store alerts in session state
            if 'alerts' not in st.session_state:
                st.session_state['alerts'] = []
            
            if st.button("Add Alert", key="add_alert"):
                alert = {
                    'ticker': ticker_input,
                    'type': alert_type,
                    'price': alert_price if alert_type in ["Price Above", "Price Below"] else None,
                    'active': True
                }
                st.session_state['alerts'].append(alert)
                st.success(f"Alert added for {ticker_input}")
            
            # Show active alerts
            if st.session_state.get('alerts'):
                st.markdown("**Active Alerts:**")
                for i, alert in enumerate(st.session_state['alerts']):
                    if alert['active']:
                        if alert['price']:
                            st.caption(f"• {alert['ticker']}: {alert['type']} ${alert['price']:.2f}")
                        else:
                            st.caption(f"• {alert['ticker']}: {alert['type']}")
                        if st.button(f"Remove", key=f"remove_alert_{i}"):
                            st.session_state['alerts'][i]['active'] = False
                            st.rerun()
    
    # Check if showing watchlist view
    if st.session_state.get('show_watchlist', False):
        show_watchlist_view()
        return
    
    # Check if showing portfolio
    if st.session_state.get('show_portfolio', False):
        show_portfolio_view()
        return
    
    # Check if showing paper trading
    if st.session_state.get('show_paper_trading', False):
        show_paper_trading_view()
        return
    
    # Check if showing backtest
    if st.session_state.get('show_backtest', False):
        show_backtest_view()
        return
    
    # Check if showing comparison
    if st.session_state.get('show_comparison', False):
        show_comparison_view()
        return
    
    # Load data
    try:
        with st.spinner(f'Loading data for {ticker_input}...'):
            df, info = load_stock_data(ticker_input, period)
            df = calculate_all_indicators(df)
        
        # Current stats
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check alerts
        if st.session_state.get('alerts'):
            triggered_alerts = []
            for alert in st.session_state['alerts']:
                if not alert['active'] or alert['ticker'] != ticker_input:
                    continue
                
                alert_triggered = False
                if alert['type'] == "Price Above" and current['Close'] > alert['price']:
                    triggered_alerts.append(f"ALERT: {alert['ticker']} is above ${alert['price']:.2f} (Current: ${current['Close']:.2f})")
                elif alert['type'] == "Price Below" and current['Close'] < alert['price']:
                    triggered_alerts.append(f"ALERT: {alert['ticker']} is below ${alert['price']:.2f} (Current: ${current['Close']:.2f})")
                elif alert['type'] == "RSI Overbought (>70)" and current['RSI'] > 70:
                    triggered_alerts.append(f"ALERT: {alert['ticker']} RSI is overbought: {current['RSI']:.1f}")
                elif alert['type'] == "RSI Oversold (<30)" and current['RSI'] < 30:
                    triggered_alerts.append(f"ALERT: {alert['ticker']} RSI is oversold: {current['RSI']:.1f}")
                elif alert['type'] == "Volume Spike (>2x avg)" and current['Volume'] > 2 * current['Volume_MA20']:
                    triggered_alerts.append(f"ALERT: {alert['ticker']} volume spike: {current['Volume']/current['Volume_MA20']:.1f}x average")
            
            if triggered_alerts:
                for alert_msg in triggered_alerts:
                    st.warning(alert_msg)
        
        # Top metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        price_change = current['Close'] - prev['Close']
        price_change_pct = (price_change / prev['Close']) * 100
        
        with col1:
            st.metric(
                label="Current Price",
                value=f"${current['Close']:.2f}",
                delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )
        
        with col2:
            st.metric(
                label="Volume",
                value=f"{current['Volume']:,.0f}",
                delta=f"{((current['Volume']/current['Volume_MA20']-1)*100):+.1f}% vs avg"
            )
        
        with col3:
            rsi_status = "Overbought" if current['RSI'] > 70 else ("Oversold" if current['RSI'] < 30 else "Normal")
            st.metric(
                label="RSI",
                value=f"{current['RSI']:.1f}",
                delta=rsi_status
            )
        
        with col4:
            st.metric(
                label="ADX (Trend)",
                value=f"{current['ADX']:.1f}",
                delta="Strong" if current['ADX'] > 25 else "Weak"
            )
        
        with col5:
            market_cap = info.get('marketCap', 0)
            if market_cap >= 1e9:
                cap_display = f"${market_cap/1e9:.2f}B"
            else:
                cap_display = f"${market_cap/1e6:.2f}M"
            st.metric(
                label="Market Cap",
                value=cap_display
            )
        
        # Company Information Section
        st.markdown("---")
        
        # Ticker and Company Name Header
        if info.get('longName'):
            st.markdown(f"## {ticker_input} - {info.get('longName')}")
        else:
            st.markdown(f"## {ticker_input}")
        
        st.markdown("---")
        
        col_company1, col_company2 = st.columns(2)
        
        with col_company1:
            with st.container():
                st.markdown("### Company Profile")
                st.metric("Sector", info.get('sector', 'N/A'))
                st.metric("Industry", info.get('industry', 'N/A'))
                if info.get('fullTimeEmployees'):
                    st.metric("Employees", f"{info.get('fullTimeEmployees'):,}")
                if info.get('city') and info.get('country'):
                    st.caption(f"Location: {info.get('city')}, {info.get('country')}")
        
        with col_company2:
            with st.container():
                st.markdown("### Analyst Coverage")
                st.caption("Source: Yahoo Finance aggregates ratings from major Wall Street firms")
                st.caption("Includes: Goldman Sachs, Morgan Stanley, JP Morgan, etc.")
                
                if 'recommendationKey' in info and info['recommendationKey']:
                    rating = info['recommendationKey'].upper()
                    if rating in ['STRONG_BUY', 'BUY']:
                        st.metric("Analyst Rating", rating, "Positive", delta_color="normal")
                    elif rating == 'HOLD':
                        st.metric("Analyst Rating", rating, "Neutral")
                    else:
                        st.metric("Analyst Rating", rating, "Negative", delta_color="inverse")
                else:
                    st.info("No analyst ratings available")
                
                if 'targetMeanPrice' in info and info['targetMeanPrice']:
                    target = info['targetMeanPrice']
                    upside = ((target - current['Close']) / current['Close']) * 100
                    st.metric(
                        "Price Target (Avg)", 
                        f"${target:.2f}", 
                        f"{upside:+.1f}% upside",
                        help="12-month price target - average of analyst forecasts"
                    )
                    
                    # Show how many analysts
                    if 'numberOfAnalystOpinions' in info:
                        st.caption(f"12-month target from {info['numberOfAnalystOpinions']} analysts")
                else:
                    st.info("No price target available")
                
                if 'trailingPE' in info and info['trailingPE']:
                    pe_val = info['trailingPE']
                    if pe_val < 15:
                        st.metric("P/E Ratio", f"{pe_val:.2f}", "Undervalued")
                    elif pe_val > 30:
                        st.metric("P/E Ratio", f"{pe_val:.2f}", "High")
                    else:
                        st.metric("P/E Ratio", f"{pe_val:.2f}", "Normal")
        
        # Get trading signal using new improved system
        params = st.session_state.get('signal_params', SignalParams())
        sig = compute_signal(df, info, params, last_label=None)
        signal = sig["label"]
        score = sig["score"]
        reasons = sig["reasons"]
        risk_info = sig["risk"]
        
        # Map signal to color for backward compatibility
        color = "green" if signal == "BUY" else ("red" if signal == "SELL" else "orange")
        
        # Get current data for specific guidance
        latest = df.iloc[-1]
        current_price = latest['Close']
        ma20 = latest['MA20']
        ma50 = latest['MA50']
        rsi = latest['RSI']
        macd = latest['MACD']
        macd_signal = latest['MACD_Signal']
        adx = latest['ADX']
        bb_lower = latest['BB_Lower']
        bb_middle = latest['BB_Middle']
        
        # Advanced support/resistance calculations
        lookback = min(60, len(df))  # Look back up to 60 days
        recent_df = df.iloc[-lookback:]
        
        # 1. Simple support (recent low)
        recent_low = recent_df['Low'].min()
        simple_support = round(recent_low * 0.99, 2)
        
        # 2. Fibonacci retracement levels
        recent_high = recent_df['High'].max()
        recent_low_fib = recent_df['Low'].min()
        fib_range = recent_high - recent_low_fib
        
        fib_236 = round(recent_high - (fib_range * 0.236), 2)  # Shallow retracement
        fib_382 = round(recent_high - (fib_range * 0.382), 2)  # Moderate retracement
        fib_500 = round(recent_high - (fib_range * 0.500), 2)  # 50% retracement
        fib_618 = round(recent_high - (fib_range * 0.618), 2)  # Golden ratio
        
        # 3. Volume-weighted support (find price level with highest volume)
        try:
            price_bins = pd.cut(recent_df['Close'], bins=20)
            volume_by_price = recent_df.groupby(price_bins)['Volume'].sum()
            high_volume_bin = volume_by_price.idxmax()
            volume_support = round((high_volume_bin.left + high_volume_bin.right) / 2, 2)
        except:
            volume_support = simple_support
        
        # 4. Previous swing lows (local minimums)
        swing_lows = []
        for i in range(2, len(recent_df) - 2):
            if (recent_df['Low'].iloc[i] < recent_df['Low'].iloc[i-1] and 
                recent_df['Low'].iloc[i] < recent_df['Low'].iloc[i-2] and
                recent_df['Low'].iloc[i] < recent_df['Low'].iloc[i+1] and
                recent_df['Low'].iloc[i] < recent_df['Low'].iloc[i+2]):
                swing_lows.append(recent_df['Low'].iloc[i])
        
        swing_support = round(min(swing_lows), 2) if swing_lows else simple_support
        
        # 5. Bollinger Band lower as dynamic support
        bb_support = round(bb_lower, 2)
        
        # Combine all support levels and rank them
        support_levels = {
            'Simple Support': simple_support,
            'Fibonacci 23.6%': fib_236,
            'Fibonacci 38.2%': fib_382,
            'Fibonacci 50%': fib_500,
            'Fibonacci 61.8%': fib_618,
            'Volume Profile': volume_support,
            'Swing Low': swing_support,
            'Bollinger Lower': bb_support,
            'MA50': round(ma50, 2)
        }
        
        # Filter to levels below current price (potential support)
        valid_supports = {k: v for k, v in support_levels.items() if v < current_price}
        
        # Sort by price (highest to lowest)
        sorted_supports = sorted(valid_supports.items(), key=lambda x: x[1], reverse=True)
        
        # Best entry is the highest support level below current (nearest support)
        # Conservative entry is the lowest (maximum safety)
        best_entry = sorted_supports[0] if sorted_supports else (None, current_price * 0.95)
        conservative_entry = sorted_supports[-1] if sorted_supports else (None, current_price * 0.90)
        
        # Also calculate resistance levels (for take profit)
        resistance_levels = {
            'Fibonacci 23.6% R': round(recent_low_fib + (fib_range * 0.236), 2),
            'Fibonacci 38.2% R': round(recent_low_fib + (fib_range * 0.382), 2),
            'MA20': round(ma20, 2),
            'BB Middle': round(bb_middle, 2),
            'Recent High': round(recent_high, 2)
        }
        
        valid_resistances = {k: v for k, v in resistance_levels.items() if v > current_price}
        sorted_resistances = sorted(valid_resistances.items(), key=lambda x: x[1])
        
        # Analysis Alignment Section
        st.markdown("---")
        st.subheader("Fundamental vs Technical Analysis")
        
        # Determine analyst sentiment
        analyst_rating = info.get('recommendationKey', '').upper()
        analyst_bullish = analyst_rating in ['STRONG_BUY', 'BUY']
        analyst_bearish = analyst_rating in ['STRONG_SELL', 'SELL']
        
        # Determine technical sentiment
        tech_bullish = signal in ["STRONG BUY", "BUY"]
        tech_bearish = signal in ["SELL", "STRONG SELL"]
        
        # Create comparison
        col_comp1, col_comp2, col_comp3 = st.columns(3)
        
        with col_comp1:
            st.markdown("### Wall Street View")
            st.markdown("**(Long-term: 6-12 months)**")
            if analyst_bullish:
                st.markdown(f"### :green[{analyst_rating}]")
                st.caption("Analysts are bullish on fundamentals")
            elif analyst_bearish:
                st.markdown(f"### :red[{analyst_rating}]")
                st.caption("Analysts are bearish on fundamentals")
            else:
                st.markdown(f"### :orange[{analyst_rating or 'N/A'}]")
                st.caption("Analysts are neutral")
        
        with col_comp2:
            st.markdown("### Technical Signal")
            st.markdown("**(Short-term: Days to weeks)**")
            if tech_bullish:
                st.markdown(f"### :green[{signal}]")
                st.caption(f"Score: {score}/100 (Bullish)")
            elif tech_bearish:
                st.markdown(f"### :red[{signal}]")
                st.caption(f"Score: {score}/100 (Bearish)")
            else:
                st.markdown(f"### :orange[{signal}]")
                st.caption(f"Score: {score}/100 (Neutral)")
            
            # Show signal reasoning
            with st.expander("Why this signal?", expanded=False):
                st.markdown("**Signal Factors:**")
                for reason in reasons[:10]:
                    st.write(reason)
                st.markdown("---")
                st.caption(f"**Risk Management:**")
                st.caption(f"Stop Loss: ${risk_info['stop']:.2f}")
                st.caption(f"First Target: ${risk_info['tgt1']:.2f}")
                st.caption(f"ATR: {risk_info['atr_pct']:.2f}%")
                if risk_info['reward_risk'] > 0:
                    st.caption(f"Reward/Risk: {risk_info['reward_risk']:.2f}:1")
        
        with col_comp3:
            st.markdown("### What This Means")
            
            # Determine alignment and provide specific guidance
            if analyst_bullish and tech_bullish:
                st.markdown("### :green[ALIGNED - BUY]")
                st.success("**Strong Buy Setup**\n\nFundamentals strong + Technicals bullish")
                
                # Specific entry guidance
                st.markdown("**Entry Strategy:**")
                st.write(f"• **Enter now at ${current_price:.2f}**")
                if sorted_supports and best_entry[1] < current_price:
                    st.write(f"• Or add more on dip to ${best_entry[1]} ({best_entry[0]})")
                st.write(f"• Stop loss: Below ${best_entry[1] if best_entry[0] else ma50:.2f}")
                
                # Show profit targets
                if info.get('targetMeanPrice'):
                    target = info['targetMeanPrice']
                    upside = ((target - current_price) / current_price) * 100
                    # Get target timeframe (typically 12 months)
                    timeframe = "12-month target"
                    st.write(f"• Analyst target: ${target:.2f} (+{upside:.1f}%)")
                    st.caption(f"  ↳ {timeframe} from {info.get('numberOfAnalystOpinions', 'multiple')} analysts")
                
                # Show near-term resistance levels
                if sorted_resistances:
                    st.markdown("**Take Profit Levels:**")
                    for i, (name, price) in enumerate(sorted_resistances[:3]):
                        profit_pct = ((price - current_price) / current_price) * 100
                        if i == 0:
                            st.caption(f"Short-term: ${price} ({name}, +{profit_pct:.1f}%)")
                        elif i == 1:
                            st.caption(f"Medium-term: ${price} ({name}, +{profit_pct:.1f}%)")
                        else:
                            st.caption(f"Swing target: ${price} ({name}, +{profit_pct:.1f}%)")
            
            elif analyst_bullish and tech_bearish:
                st.markdown("### :orange[MIXED - WAIT]")
                st.warning("**Wait for Better Entry**\n\nFundamentals strong but technicals weak")
                
                # Specific entry guidance
                st.markdown("**Entry Strategy:**")
                
                # Show momentum triggers
                st.markdown("**Option 1: Wait for Momentum Confirmation**")
                momentum_triggers = []
                
                if current_price < ma20:
                    momentum_triggers.append(f"• Price breaks above ${ma20:.2f} (MA20)")
                
                if macd < macd_signal:
                    momentum_triggers.append(f"• MACD crosses above signal line")
                
                if rsi < 40:
                    momentum_triggers.append(f"• RSI bounces from oversold ({rsi:.1f} to 40+)")
                elif rsi > 60:
                    momentum_triggers.append(f"• RSI cools to 40-50 (currently {rsi:.1f})")
                else:
                    momentum_triggers.append(f"• RSI remains healthy (currently {rsi:.1f})")
                
                for trigger in momentum_triggers:
                    st.write(trigger)
                
                # Show support level entries
                st.markdown("**Option 2: Buy at Support Levels**")
                st.caption("Multiple technical support levels identified:")
                
                # Show top 3-4 support levels
                for i, (name, price) in enumerate(sorted_supports[:4]):
                    if info.get('targetMeanPrice'):
                        target = info['targetMeanPrice']
                        upside = ((target - price) / price) * 100
                        
                        if i == 0:
                            st.write(f"**${price} - {name}** (nearest, +{upside:.1f}% to target)")
                        elif i == len(sorted_supports) - 1 or i == 3:
                            st.write(f"**${price} - {name}** (conservative, +{upside:.1f}% to target)")
                        else:
                            st.write(f"${price} - {name} (+{upside:.1f}% to target)")
                    else:
                        if i == 0:
                            st.write(f"**${price} - {name}** (nearest support)")
                        else:
                            st.write(f"${price} - {name}")
                
                # Recommendation
                if best_entry[0]:
                    st.markdown(f"**Recommended:** Set limit orders at **${best_entry[1]}** ({best_entry[0]})")
                    if conservative_entry[0] and conservative_entry[1] != best_entry[1]:
                        st.caption(f"Conservative: ${conservative_entry[1]} ({conservative_entry[0]})")
                
                # Risk/Reward comparison
                if info.get('targetMeanPrice'):
                    target = info['targetMeanPrice']
                    upside_now = ((target - current_price) / current_price) * 100
                    upside_best = ((target - best_entry[1]) / best_entry[1]) * 100
                    upside_conservative = ((target - conservative_entry[1]) / conservative_entry[1]) * 100
                    
                    st.markdown("**Risk/Reward Analysis:**")
                    st.caption(f"Buy now at ${current_price:.2f}: +{upside_now:.1f}% to target")
                    st.caption(f"Buy at ${best_entry[1]}: +{upside_best:.1f}% to target (adds +{upside_best-upside_now:.1f}%)")
                    st.caption(f"Buy at ${conservative_entry[1]}: +{upside_conservative:.1f}% to target (adds +{upside_conservative-upside_now:.1f}%)")
            
            elif analyst_bearish and tech_bullish:
                st.markdown("### :orange[MIXED - CAUTION]")
                st.warning("**Trade Only, Don't Invest**\n\nTechnicals bullish but fundamentals weak")
                
                # Specific exit guidance
                st.markdown("**If Trading (Short-term Only):**")
                st.write(f"• Entry: ${current_price:.2f} (current)")
                st.write(f"• Tight stop: ${current_price * 0.97:.2f} (-3%)")
                
                # Show quick exit targets
                if sorted_resistances:
                    st.markdown("**Quick Exit Targets:**")
                    for i, (name, price) in enumerate(sorted_resistances[:2]):
                        profit_pct = ((price - current_price) / current_price) * 100
                        st.write(f"• ${price} - {name} (+{profit_pct:.1f}%)")
                else:
                    st.write(f"• Take profit at: ${ma20:.2f} (MA20)")
                
                st.caption("Exit immediately if fundamentals worsen or technical signal flips")
            
            elif analyst_bearish and tech_bearish:
                st.markdown("### :red[ALIGNED - AVOID]")
                st.error("**Stay Away**\n\nBoth fundamentals and technicals negative")
                
                st.markdown("**Why Avoid:**")
                st.write(f"• Analysts bearish: {analyst_rating}")
                st.write(f"• Technical signal: {signal} ({score}/5)")
                st.write(f"• Downtrend likely to continue")
                st.write(f"• Wait for both to improve")
            
            else:
                st.markdown("### :grey[NEUTRAL]")
                st.info("**No Clear Signal**\n\nMixed or neutral signals")
                
                st.markdown("**Wait For:**")
                st.write(f"• Clear trend to develop")
                st.write(f"• Analyst rating to change")
                st.write(f"• Technical conditions to align")
                st.write(f"• Better opportunity elsewhere")
        
        # Position Sizing Calculator
        st.markdown("---")
        st.subheader("Position Sizing Calculator")
        
        with st.expander("Calculate Position Size", expanded=False):
            st.markdown("**Risk Management Tool** - Calculate how many shares to buy based on your risk tolerance")
            
            col_ps1, col_ps2, col_ps3 = st.columns(3)
            
            with col_ps1:
                portfolio_capital = st.number_input(
                    "Portfolio Value ($)",
                    min_value=100.0,
                    value=10000.0,
                    step=100.0,
                    help="Total capital available for trading"
                )
                risk_pct = st.slider(
                    "Risk per Trade (%)",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                    help="% of portfolio you're willing to risk on this trade (2% is common)"
                )
            
            with col_ps2:
                entry_price = st.number_input(
                    "Entry Price ($)",
                    min_value=0.01,
                    value=float(current_price),
                    step=0.01,
                    help="Price you plan to enter at"
                )
                stop_price = st.number_input(
                    "Stop Loss ($)",
                    min_value=0.01,
                    value=float(risk_info['stop']),
                    step=0.01,
                    help="Price at which you'll exit to limit losses"
                )
            
            with col_ps3:
                # Calculate position
                pos_calc = position_size_calculator(portfolio_capital, risk_pct, entry_price, stop_price)
                
                st.metric("Shares to Buy", f"{pos_calc['shares']:,}")
                st.metric("Position Value", f"${pos_calc['position_value']:,.2f}")
                st.metric("$ at Risk", f"${pos_calc['risk_amount']:.2f}")
                
                # Risk/Reward if target set
                if info.get('targetMeanPrice'):
                    target_price = info['targetMeanPrice']
                    potential_gain = (target_price - entry_price) * pos_calc['shares']
                    rr_ratio = potential_gain / pos_calc['risk_amount'] if pos_calc['risk_amount'] > 0 else 0
                    st.metric("Reward/Risk", f"{rr_ratio:.2f}:1")
                    st.caption(f"Potential gain to analyst target: ${potential_gain:,.2f}")
            
            # Add explanation
            st.markdown("---")
            st.info(f"""
            **How it works:**
            - You're risking **{risk_pct}%** of your ${portfolio_capital:,.0f} portfolio = **${portfolio_capital * risk_pct / 100:,.2f}**
            - Risk per share: ${entry_price - stop_price:.2f} (entry - stop)
            - Shares: ${portfolio_capital * risk_pct / 100:,.2f} ÷ ${entry_price - stop_price:.2f} = **{pos_calc['shares']} shares**
            - If stopped out, you lose ${pos_calc['risk_amount']:.2f} ({risk_pct}% of portfolio)
            """)
        
        # Earnings Analysis Section
        st.markdown("---")
        st.subheader("Earnings Analysis")
        
        try:
            with st.spinner('Loading earnings data...'):
                earnings_data = load_earnings_data(ticker_input)
            
            # Create tabs for different earnings views
            tab1, tab2, tab3 = st.tabs(["EPS Trends", "Earnings Estimates", "Revenue History"])
            
            with tab1:
                # EPS Trend Chart
                eps_chart = create_eps_chart(earnings_data)
                if eps_chart:
                    st.plotly_chart(eps_chart, width='stretch')
                    
                    # Show recent earnings data
                    earnings_df = earnings_data.get('earnings_df')
                    if earnings_df is not None and not earnings_df.empty:
                        st.write("**Recent Quarterly Earnings:**")
                        # Get last 4 quarters
                        recent = earnings_df.iloc[:, -4:] if earnings_df.shape[1] > 4 else earnings_df
                        # Transpose for display (quarters as rows)
                        display_df = recent.T
                        # Format the quarter labels
                        display_df.index = [col.strftime('%Y-%m') if hasattr(col, 'strftime') else str(col) 
                                          for col in display_df.index]
                        display_df.index.name = 'Quarter'
                        
                        # Format columns
                        format_dict = {}
                        if 'Revenue' in display_df.columns:
                            format_dict['Revenue'] = '${:,.0f}'
                        if 'Net Income' in display_df.columns:
                            format_dict['Net Income'] = '${:,.0f}'
                        if 'EPS' in display_df.columns:
                            format_dict['EPS'] = '${:.2f}'
                        
                        st.dataframe(display_df.style.format(format_dict), width='stretch')
                else:
                    st.info("No historical earnings data available for this ticker")
            
            with tab2:
                # Earnings Estimates Table
                st.write("**Analyst Earnings Estimates**")
                st.caption("Source: Yahoo Finance aggregated analyst forecasts")
                
                # Try to get earnings estimate data from info
                stock = yf.Ticker(ticker_input)
                
                # Create estimates table
                estimates_data = []
                
                # Try to get from earnings_dates or earnings_estimate
                if hasattr(stock, 'earnings_dates') and stock.earnings_dates is not None and len(stock.earnings_dates) > 0:
                    earnings_dates_df = stock.earnings_dates
                    
                    # Get unique periods
                    if 'EPS Estimate' in earnings_dates_df.columns and 'Reported EPS' in earnings_dates_df.columns:
                        for idx, row in earnings_dates_df.head(4).iterrows():
                            period_name = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                            estimates_data.append({
                                'Period': period_name,
                                'EPS Estimate': row.get('EPS Estimate', 'N/A'),
                                'Reported EPS': row.get('Reported EPS', 'N/A')
                            })
                
                if estimates_data:
                    estimates_df = pd.DataFrame(estimates_data)
                    st.dataframe(estimates_df, width='stretch')
                else:
                    # Show a simplified view using info data
                    col_est1, col_est2, col_est3 = st.columns(3)
                    
                    with col_est1:
                        st.markdown("**Current Quarter**")
                        if 'numberOfAnalystOpinions' in info:
                            st.metric("Analysts Covering", f"{info['numberOfAnalystOpinions']}")
                        if 'forwardEps' in info and info['forwardEps']:
                            st.metric("Forward EPS", f"${info['forwardEps']:.2f}")
                    
                    with col_est2:
                        st.markdown("**Valuation**")
                        if 'trailingEps' in info and info['trailingEps']:
                            st.metric("Trailing EPS", f"${info['trailingEps']:.2f}")
                        if 'trailingPE' in info and info['trailingPE']:
                            st.metric("P/E Ratio", f"{info['trailingPE']:.2f}")
                    
                    with col_est3:
                        st.markdown("**Growth**")
                        if 'earningsGrowth' in info and info['earningsGrowth']:
                            growth = info['earningsGrowth'] * 100
                            st.metric("Earnings Growth", f"{growth:+.1f}%")
                        if 'revenueGrowth' in info and info['revenueGrowth']:
                            rev_growth = info['revenueGrowth'] * 100
                            st.metric("Revenue Growth", f"{rev_growth:+.1f}%")
            
            with tab3:
                # Revenue vs Earnings Chart
                rev_chart = create_revenue_earnings_chart(earnings_data)
                if rev_chart:
                    st.plotly_chart(rev_chart, width='stretch')
                    
                    # Show revenue metrics
                    col_rev1, col_rev2, col_rev3 = st.columns(3)
                    
                    with col_rev1:
                        if 'totalRevenue' in info and info['totalRevenue']:
                            # Use same logic as chart - millions unless over 1B
                            revenue = info['totalRevenue']
                            if revenue >= 1e9:
                                st.metric(
                                    "Total Revenue",
                                    f"${revenue/1e9:.2f}B"
                                )
                            else:
                                st.metric(
                                    "Total Revenue",
                                    f"${revenue/1e6:.2f}M"
                                )
                    
                    with col_rev2:
                        if 'revenueGrowth' in info and info['revenueGrowth']:
                            st.metric(
                                "Revenue Growth",
                                f"{info['revenueGrowth']*100:+.1f}%"
                            )
                    
                    with col_rev3:
                        if 'grossMargins' in info and info['grossMargins']:
                            st.metric(
                                "Gross Margin",
                                f"{info['grossMargins']*100:.1f}%"
                            )
                else:
                    st.info("No revenue history data available for this ticker")
        
        except Exception as e:
            st.warning(f"Could not load earnings data: {str(e)}")
            st.info("Some tickers may not have complete earnings data available")
        
        # Enhanced Valuation Analysis Section
        st.markdown("---")
        st.subheader("Valuation Analysis")
        st.caption("Multi-factor valuation using DCF, P/S analysis, and fundamental scoring")
        
        try:
            with st.spinner('Analyzing valuation...'):
                val_analyzer = EnhancedValuation(ticker_input)
                valuation = val_analyzer.get_comprehensive_valuation()
            
            # Overall valuation status banner
            overall_status = valuation['overall_status']
            if overall_status == 'OVERVALUED':
                st.error(f"**{overall_status}** - Stock appears expensive relative to fundamentals")
            elif overall_status == 'UNDERVALUED':
                st.success(f"**{overall_status}** - Stock may be trading below fair value")
            elif overall_status == 'FAIR_VALUE':
                st.info(f"**{overall_status}** - Mixed signals, stock appears fairly valued")
            else:
                st.warning(f"**{overall_status}** - Insufficient data for valuation")
            
            # Create tabs for different valuation methods
            val_tab1, val_tab2, val_tab3 = st.tabs(["DCF Analysis", "P/S Ratio", "Valuation Score"])
            
            # TAB 1: DCF Analysis
            with val_tab1:
                dcf = valuation['dcf']
                if dcf:
                    st.markdown("### Discounted Cash Flow (DCF)")
                    st.caption("Estimates intrinsic value by projecting future cash flows")
                    
                    col_dcf1, col_dcf2, col_dcf3 = st.columns(3)
                    
                    with col_dcf1:
                        st.metric(
                            "DCF Fair Value",
                            f"${abs(dcf['dcf_per_share']):.2f}",
                            help="Estimated intrinsic value per share"
                        )
                    
                    with col_dcf2:
                        st.metric(
                            "Current Price",
                            f"${dcf['current_price']:.2f}"
                        )
                    
                    with col_dcf3:
                        over_pct = dcf['overvaluation_pct']
                        delta_color = "inverse" if over_pct > 0 else "normal"
                        st.metric(
                            "Valuation Gap",
                            f"{abs(over_pct):.1f}%",
                            delta=dcf['valuation_status'],
                            delta_color=delta_color,
                            help="Positive = overvalued, Negative = undervalued"
                        )
                    
                    # Explanation
                    st.markdown("---")
                    if dcf['valuation_status'] == 'OVERVALUED':
                        st.warning(f"""
                        **DCF Analysis suggests OVERVALUATION**
                        
                        The current market price (${dcf['current_price']:.2f}) is trading **{abs(dcf['overvaluation_pct']):.1f}% above** 
                        the estimated fair value (${abs(dcf['dcf_per_share']):.2f}) based on projected cash flows.
                        
                        This may indicate the market has priced in very optimistic growth expectations.
                        """)
                    else:
                        st.success(f"""
                        **DCF Analysis suggests UNDERVALUATION**
                        
                        The current market price (${dcf['current_price']:.2f}) is trading **{abs(dcf['overvaluation_pct']):.1f}% below** 
                        the estimated fair value (${abs(dcf['dcf_per_share']):.2f}) based on projected cash flows.
                        
                        This may represent a buying opportunity if fundamentals are sound.
                        """)
                    
                    # Current FCF
                    fcf_millions = dcf['current_fcf'] / 1e6
                    if fcf_millions < 0:
                        st.info(f"**Current Free Cash Flow:** ${abs(fcf_millions):.1f}M (negative) - Typical for growth-stage biotech investing heavily in R&D")
                    else:
                        st.success(f"**Current Free Cash Flow:** ${fcf_millions:.1f}M (positive) - Company generating cash")
                
                else:
                    st.info("DCF analysis not available - insufficient cash flow data")
            
            # TAB 2: P/S Analysis
            with val_tab2:
                ps = valuation['ps_analysis']
                if ps:
                    st.markdown("### Price-to-Sales (P/S) Ratio Analysis")
                    st.caption("Compares valuation to revenue - useful for unprofitable growth companies")
                    
                    col_ps1, col_ps2, col_ps3 = st.columns(3)
                    
                    with col_ps1:
                        ps_color = "inverse" if ps['ps_ratio'] > ps['industry_avg'] else "normal"
                        st.metric(
                            f"{ticker_input} P/S",
                            f"{ps['ps_ratio']:.1f}x",
                            delta=f"vs {ps['industry_avg']:.1f}x industry",
                            delta_color=ps_color
                        )
                    
                    with col_ps2:
                        st.metric(
                            "Industry Avg",
                            f"{ps['industry_avg']:.1f}x",
                            help="Biotech industry average P/S ratio"
                        )
                    
                    with col_ps3:
                        st.metric(
                            "Peer Avg",
                            f"{ps['peer_avg']:.1f}x",
                            help="Peer group average P/S ratio"
                        )
                    
                    # Visualization
                    st.markdown("---")
                    import plotly.graph_objects as go
                    
                    fig_ps = go.Figure(go.Bar(
                        x=['Industry Avg', 'Peer Avg', ticker_input],
                        y=[ps['industry_avg'], ps['peer_avg'], ps['ps_ratio']],
                        marker_color=['#64B5F6', '#81C784', '#EF5350' if ps['is_overvalued'] else '#66BB6A'],
                        text=[f"{ps['industry_avg']:.1f}x", f"{ps['peer_avg']:.1f}x", f"{ps['ps_ratio']:.1f}x"],
                        textposition='outside'
                    ))
                    
                    fig_ps.update_layout(
                        title='P/S Ratio Comparison',
                        yaxis_title='P/S Ratio (x)',
                        height=350,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_ps, use_container_width=True)
                    
                    # Explanation
                    if ps['is_overvalued']:
                        st.warning(f"""
                        **P/S Analysis suggests OVERVALUATION**
                        
                        {ticker_input}'s P/S ratio of **{ps['ps_ratio']:.1f}x** is significantly higher than:
                        - Industry average: {ps['industry_avg']:.1f}x
                        - Peer average: {ps['peer_avg']:.1f}x
                        
                        Investors are paying a premium for each dollar of revenue, which may be justified 
                        by superior growth prospects or margins.
                        """)
                    else:
                        st.success(f"""
                        **P/S Analysis suggests FAIR/UNDERVALUATION**
                        
                        {ticker_input}'s P/S ratio of **{ps['ps_ratio']:.1f}x** is in line with or below:
                        - Industry average: {ps['industry_avg']:.1f}x
                        - Peer average: {ps['peer_avg']:.1f}x
                        
                        The stock appears reasonably valued relative to its revenue.
                        """)
                
                else:
                    st.info("P/S analysis not available - insufficient revenue data")
            
            # TAB 3: Valuation Score
            with val_tab3:
                score_data = valuation['valuation_score']
                
                st.markdown("### Multi-Factor Valuation Score")
                st.caption("Comprehensive scoring across 6 fundamental factors")
                
                # Score display
                col_score1, col_score2 = st.columns([1, 2])
                
                with col_score1:
                    score_pct = (score_data['score'] / score_data['max_score']) * 100
                    
                    st.markdown(f"## {score_data['score']}/{score_data['max_score']}")
                    st.markdown(f"### Grade: **{score_data['grade']}**")
                    
                    # Progress bar
                    st.progress(score_pct / 100)
                
                with col_score2:
                    st.markdown("**Valuation Factors:**")
                    for factor in score_data['factors']:
                        if '✓' in factor:
                            st.markdown(f"- PASS: {factor.replace('✓ ', '')}")
                        else:
                            st.markdown(f"- FAIL: {factor.replace('✗ ', '')}")
                
                # Interpretation
                st.markdown("---")
                if score_data['score'] >= 5:
                    st.success("""
                    **Excellent Fundamentals** (Grade A)
                    
                    This stock checks most valuation boxes. Strong fundamentals suggest it may be undervalued 
                    or fairly priced. However, always consider growth prospects and industry context.
                    """)
                elif score_data['score'] >= 4:
                    st.success("""
                    **Good Fundamentals** (Grade B)
                    
                    This stock has solid fundamentals with room for improvement. Generally a positive sign 
                    for long-term value investors.
                    """)
                elif score_data['score'] >= 3:
                    st.info("""
                    **Mixed Fundamentals** (Grade C)
                    
                    This stock shows average fundamentals. Some strengths, some weaknesses. Requires deeper 
                    analysis to determine if it's a good fit for your strategy.
                    """)
                else:
                    st.warning("""
                    **Weak Fundamentals** (Grade D/F)
                    
                    This stock fails most valuation checks. Common for early-stage growth companies, 
                    especially in biotech. High risk, high potential reward scenario.
                    """)
                
                # Factor explanations
                with st.expander("What do these factors mean?"):
                    st.markdown("""
                    **P/E Ratio** - Price relative to earnings (PASS if < 15)  
                    **P/S Ratio** - Price relative to sales vs industry (PASS if below average)  
                    **Debt-to-Equity** - Financial leverage (PASS if < 0.5)  
                    **Current Ratio** - Short-term liquidity (PASS if > 2)  
                    **Revenue Growth** - Top-line growth (PASS if > 10%)  
                    **Profitability** - Profit margins (PASS if > 10%)  
                    """)
        
        except Exception as e:
            st.warning(f"Could not load valuation analysis: {str(e)}")
            st.info("Valuation analysis may not be available for all tickers")
        
        # News & Sentiment Section
        st.markdown("---")
        st.subheader("News & Sentiment Analysis")
        try:
            with st.spinner('Loading news...'):
                news_analyzer = NewsAnalyzer(ticker_input)
                news_summary = news_analyzer.get_news_summary(limit=15)
                news_df = news_analyzer.get_news_with_sentiment(limit=15)
            
            # Sentiment Summary
            col_sent1, col_sent2, col_sent3, col_sent4 = st.columns(4)
            
            with col_sent1:
                st.metric("Total Articles", news_summary['total_articles'])
            with col_sent2:
                st.metric("Bullish", f"{news_summary['bullish']} ({news_summary['bullish_pct']:.0f}%)")
            with col_sent3:
                st.metric("Bearish", f"{news_summary['bearish']} ({news_summary['bearish_pct']:.0f}%)")
            with col_sent4:
                trend = news_summary['sentiment_trend']
                if trend == 'BULLISH':
                    st.success(f"**Trend: {trend}**")
                elif trend == 'BEARISH':
                    st.error(f"**Trend: {trend}**")
                else:
                    st.info(f"**Trend: {trend}**")
            
            # News Feed
            if not news_df.empty:
                st.markdown("### Recent Headlines")
                
                # Filter options
                sentiment_filter = st.selectbox(
                    "Filter by sentiment",
                    ["All", "Bullish", "Bearish", "Neutral"],
                    key="sentiment_filter"
                )
                
                # Apply filter
                if sentiment_filter != "All":
                    filtered_df = news_df[news_df['sentiment'] == sentiment_filter.upper()]
                else:
                    filtered_df = news_df
                
                # Display news articles
                for idx, article in filtered_df.iterrows():
                    with st.expander(f"[{article['time_ago']}] {article['title']}", expanded=False):
                        col_news1, col_news2 = st.columns([3, 1])
                        
                        with col_news1:
                            st.caption(f"**Source:** {article['publisher']}")
                            st.caption(f"**Published:** {article['published'].strftime('%Y-%m-%d %H:%M')}")
                            if article['link']:
                                st.markdown(f"[Read full article]({article['link']})")
                        
                        with col_news2:
                            sentiment = article['sentiment']
                            if sentiment == 'BULLISH':
                                st.success(f"**{sentiment}**")
                            elif sentiment == 'BEARISH':
                                st.error(f"**{sentiment}**")
                            else:
                                st.info(f"**{sentiment}**")
                            st.caption(f"Confidence: {article['confidence']:.2f}")
            else:
                st.info("No recent news found for this ticker")
        
        except Exception as e:
            st.error(f"Could not load news: {str(e)}")
            st.write(f"Error type: {type(e).__name__}")
            import traceback
            st.code(traceback.format_exc())
        
        # Sector Analysis Section
        st.markdown("---")
        st.subheader("Sector & Peer Analysis")
        
        try:
            with st.spinner('Analyzing sector performance...'):
                sector_analyzer = SectorAnalyzer(ticker_input)
                sector_info = sector_analyzer.get_sector_info()
                sector_summary = sector_analyzer.get_sector_summary(period='1y')
                peer_df = sector_analyzer.compare_performance(period='1y')
            
            # Sector Info
            col_sec1, col_sec2, col_sec3 = st.columns(3)
            
            with col_sec1:
                st.markdown("**Sector Information**")
                st.write(f"Sector: {sector_info['sector']}")
                st.write(f"Industry: {sector_info['industry']}")
            
            with col_sec2:
                st.markdown("**Peer Group**")
                if sector_info['peers']:
                    st.write(", ".join(sector_info['peers']))
                else:
                    st.write("No peers defined")
            
            with col_sec3:
                st.markdown("**Comparison Period**")
                st.write("1 Year Performance")
            
            # Performance vs Peers
            if sector_summary:
                st.markdown("### Performance vs Peers (1Y)")
                
                col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
                
                with col_perf1:
                    return_color = "normal" if sector_summary['return'] >= 0 else "inverse"
                    st.metric(f"{ticker_input} Return", f"{sector_summary['return']:.2f}%", delta_color=return_color)
                
                with col_perf2:
                    st.metric("Peer Avg Return", f"{sector_summary['peer_avg_return']:.2f}%")
                
                with col_perf3:
                    outperf_color = "normal" if sector_summary['outperformance'] >= 0 else "inverse"
                    st.metric("Outperformance", f"{sector_summary['outperformance']:+.2f}%", delta_color=outperf_color)
                
                with col_perf4:
                    rank_display = f"#{sector_summary['rank']} of {sector_summary['total_stocks']}"
                    st.metric("Rank", rank_display)
                
                # Relative Strength gauge
                rel_strength = sector_summary['relative_strength']
                st.markdown(f"**Relative Strength:** {rel_strength:.1f}/100")
                st.progress(rel_strength / 100)
                
                if rel_strength >= 70:
                    st.success("Strong relative strength - outperforming most peers")
                elif rel_strength <= 30:
                    st.error("Weak relative strength - underperforming most peers")
                else:
                    st.info("Moderate relative strength")
            
            # Peer Comparison Table
            if not peer_df.empty:
                st.markdown("### Peer Comparison")
                
                # Prepare display dataframe
                display_df = peer_df.copy()
                display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: f"${x/1e9:.2f}B" if x > 0 else "N/A")
                display_df['Return'] = display_df['Return'].apply(lambda x: f"{x:+.2f}%")
                display_df['Volatility'] = display_df['Volatility'].apply(lambda x: f"{x:.2f}%")
                display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
                display_df['Max Drawdown'] = display_df['Max Drawdown'].apply(lambda x: f"{x:.2f}%")
                
                # Highlight target stock
                def highlight_target(row):
                    if row['Is Target']:
                        return ['background-color: rgba(0, 180, 216, 0.2)'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    display_df[['Ticker', 'Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Market Cap']]
                        .style.apply(highlight_target, axis=1),
                    use_container_width=True
                )
        
        except Exception as e:
            st.warning(f"Could not load sector analysis: {str(e)}")
        
        # Trading Signal (already calculated above for comparison)
        st.markdown("---")
        col_signal1, col_signal2 = st.columns([1, 2])
        
        with col_signal1:
            if color == "green":
                st.markdown(f'<p class="buy-signal">Signal: {signal}</p>', unsafe_allow_html=True)
            elif color == "red":
                st.markdown(f'<p class="sell-signal">Signal: {signal}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="neutral-signal">Signal: {signal}</p>', unsafe_allow_html=True)
            
            st.write(f"**Score: {score}/100**")
        
        with col_signal2:
            st.write("**Signal Analysis:**")
            
            # Show reasons from new signal system
            for reason in reasons[:8]:
                if reason.startswith("[+]"):
                    st.markdown(f":green[{reason}]")
                elif reason.startswith("[-]"):
                    st.markdown(f":red[{reason}]")
                else:
                    st.markdown(f":orange[{reason}]")
            
            st.markdown("---")
            
            # Add contextual explanation based on signal and score
            if signal == "BUY" and score >= 70:
                st.success(f"""
                    **Strong Buy Opportunity**
                    
                    Score: {score}/100 - Multiple confirming signals
                    
                    **Why BUY?** Technical indicators show bullish momentum with healthy conditions.
                """)
            elif signal == "BUY":
                st.info(f"""
                    **Decent Buy Setup**
                    
                    Score: {score}/100
                    
                    **Why BUY?** Core trend and momentum are positive.
                """)
            elif signal == "HOLD" and score >= 50:
                st.warning(f"""
                    **Mixed Signals - HOLD**
                    
                    Score: {score}/100 - Neither strongly bullish nor bearish
                    
                    **Action:** Wait for clearer trend.
                """)
            elif signal == "HOLD":
                st.warning(f"""
                    **Weak Setup - HOLD**
                    
                    Score: {score}/100
                    
                    **Action:** Monitor for improvement.
                """)
            elif signal == "SELL":
                st.error(f"""
                    **Bearish Signal - SELL**
                    
                    Score: {score}/100
                    
                    **Action:** Avoid new entries.
                """)
        
        # Main chart
        st.markdown("---")
        st.subheader("Technical Analysis Chart")
        fig = create_price_chart(df, ticker_input)
        st.plotly_chart(fig, width='stretch')
        
        # Detailed indicators
        st.markdown("---")
        st.subheader("Detailed Technical Indicators")
        
        # Create styled metric cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.expander("Moving Averages", expanded=True):
                ma20_diff = ((current['Close'] - current['MA20']) / current['MA20']) * 100
                ma50_diff = ((current['Close'] - current['MA50']) / current['MA50']) * 100
                
                st.metric("MA20", f"${current['MA20']:.2f}", f"{ma20_diff:+.2f}%")
                st.metric("MA50", f"${current['MA50']:.2f}", f"{ma50_diff:+.2f}%")
                if not pd.isna(current['MA200']):
                    ma200_diff = ((current['Close'] - current['MA200']) / current['MA200']) * 100
                    st.metric("MA200", f"${current['MA200']:.2f}", f"{ma200_diff:+.2f}%")
            
            with st.expander("Bollinger Bands", expanded=True):
                bb_position = ((current['Close'] - current['BB_Lower']) / 
                              (current['BB_Upper'] - current['BB_Lower']) * 100)
                
                st.metric("Upper Band", f"${current['BB_Upper']:.2f}")
                st.metric("Middle Band", f"${current['BB_Middle']:.2f}")
                st.metric("Lower Band", f"${current['BB_Lower']:.2f}")
                st.progress(bb_position / 100, text=f"Position: {bb_position:.1f}%")
        
        with col2:
            with st.expander("Momentum Indicators", expanded=True):
                # RSI with color
                rsi_val = current['RSI']
                if rsi_val > 70:
                    st.metric("RSI", f"{rsi_val:.1f}", "Overbought", delta_color="inverse")
                elif rsi_val < 30:
                    st.metric("RSI", f"{rsi_val:.1f}", "Oversold", delta_color="normal")
                else:
                    st.metric("RSI", f"{rsi_val:.1f}", "Normal")
                
                # Stochastic with color
                stoch_val = current['Stochastic']
                if stoch_val > 80:
                    st.metric("Stochastic", f"{stoch_val:.1f}", "Overbought", delta_color="inverse")
                elif stoch_val < 20:
                    st.metric("Stochastic", f"{stoch_val:.1f}", "Oversold", delta_color="normal")
                else:
                    st.metric("Stochastic", f"{stoch_val:.1f}", "Normal")
                
                # MACD
                macd_signal = "Bullish" if current['MACD'] > current['MACD_Signal'] else "Bearish"
                macd_color = "normal" if current['MACD'] > current['MACD_Signal'] else "inverse"
                st.metric("MACD", f"{current['MACD']:.3f}", macd_signal, delta_color=macd_color)
                st.caption(f"Signal Line: {current['MACD_Signal']:.3f}")
            
            with st.expander("Trend Strength", expanded=True):
                # ADX with interpretation
                adx_val = current['ADX']
                if adx_val > 40:
                    adx_status = "Very Strong"
                elif adx_val > 25:
                    adx_status = "Strong"
                elif adx_val > 20:
                    adx_status = "Moderate"
                else:
                    adx_status = "Weak"
                
                st.metric("ADX", f"{adx_val:.1f}", adx_status)
                
                # Directional indicators
                if current['Plus_DI'] > current['Minus_DI']:
                    st.metric("+DI (Bullish)", f"{current['Plus_DI']:.1f}", "Dominant", delta_color="normal")
                    st.metric("-DI (Bearish)", f"{current['Minus_DI']:.1f}")
                else:
                    st.metric("+DI (Bullish)", f"{current['Plus_DI']:.1f}")
                    st.metric("-DI (Bearish)", f"{current['Minus_DI']:.1f}", "Dominant", delta_color="inverse")
        
        with col3:
            with st.expander("Volatility", expanded=True):
                atr_pct = (current['ATR'] / current['Close']) * 100
                
                if atr_pct > 5:
                    vol_status = "High"
                elif atr_pct > 3:
                    vol_status = "Moderate"
                else:
                    vol_status = "Low"
                
                st.metric("ATR", f"${current['ATR']:.2f}", vol_status)
                st.metric("ATR %", f"{atr_pct:.2f}%")
            
            with st.expander("Volume Analysis", expanded=True):
                vol_ratio = current['Volume'] / current['Volume_MA20']
                
                st.metric("Today's Volume", f"{current['Volume']:,.0f}")
                st.metric("Avg Volume (20d)", f"{current['Volume_MA20']:,.0f}")
                
                if vol_ratio > 1.5:
                    st.metric("Volume Ratio", f"{vol_ratio:.2f}x", "High", delta_color="normal")
                elif vol_ratio < 0.5:
                    st.metric("Volume Ratio", f"{vol_ratio:.2f}x", "Low", delta_color="inverse")
                else:
                    st.metric("Volume Ratio", f"{vol_ratio:.2f}x", "Normal")
                
                obv_change = ((current['OBV'] - df['OBV'].iloc[-20]) / df['OBV'].iloc[-20]) * 100
                if obv_change > 10:
                    st.metric("OBV Change (20d)", f"{obv_change:+.2f}%", "Strong Buying", delta_color="normal")
                elif obv_change < -10:
                    st.metric("OBV Change (20d)", f"{obv_change:+.2f}%", "Strong Selling", delta_color="inverse")
                else:
                    st.metric("OBV Change (20d)", f"{obv_change:+.2f}%", "Neutral")
        
        # Footer
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data delayed 15-30 min (yfinance free tier)")
        
    except Exception as e:
        st.error(f"Error loading data for {ticker_input}: {str(e)}")
        st.info("Please check if the ticker symbol is correct and try again.")

if __name__ == "__main__":
    main()

