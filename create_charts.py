"""
Advanced Stock Charting with Technical Indicators
Creates professional charts with all indicators visualized
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our indicators
import sys
sys.path.append('.')
from advanced_indicators import TechnicalIndicators

class StockCharts:
    """Create professional stock charts with indicators"""
    
    def __init__(self, ticker, period='6mo'):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.df = self.stock.history(period=period, interval='1d')
        self.calculate_all_indicators()
    
    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        df = self.df
        
        # Moving averages
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
        
        # Volume indicators
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
        
        self.df = df
    
    def create_comprehensive_chart(self, save_path=None):
        """
        Create comprehensive multi-panel chart
        Panel 1: Price + MAs + Bollinger Bands
        Panel 2: RSI
        Panel 3: MACD
        Panel 4: Volume + OBV
        Panel 5: Stochastic
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1], hspace=0.3)
        
        # Panel 1: Price Chart with MAs and Bollinger Bands
        ax1 = plt.subplot(gs[0])
        ax1.plot(self.df.index, self.df['Close'], label='Close', color='black', linewidth=2)
        ax1.plot(self.df.index, self.df['MA20'], label='MA20', color='blue', alpha=0.7)
        ax1.plot(self.df.index, self.df['MA50'], label='MA50', color='orange', alpha=0.7)
        if not self.df['MA200'].isna().all():
            ax1.plot(self.df.index, self.df['MA200'], label='MA200', color='red', alpha=0.7)
        
        # Bollinger Bands
        ax1.plot(self.df.index, self.df['BB_Upper'], 'g--', alpha=0.3, label='BB Upper')
        ax1.plot(self.df.index, self.df['BB_Lower'], 'r--', alpha=0.3, label='BB Lower')
        ax1.fill_between(self.df.index, self.df['BB_Upper'], self.df['BB_Lower'], alpha=0.1, color='gray')
        
        ax1.set_title(f'{self.ticker} - Price Chart with Technical Indicators', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: RSI
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(self.df.index, self.df['RSI'], label='RSI', color='purple', linewidth=1.5)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax2.fill_between(self.df.index, 70, 100, alpha=0.1, color='red')
        ax2.fill_between(self.df.index, 0, 30, alpha=0.1, color='green')
        ax2.set_ylabel('RSI', fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: MACD
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.plot(self.df.index, self.df['MACD'], label='MACD', color='blue', linewidth=1.5)
        ax3.plot(self.df.index, self.df['MACD_Signal'], label='Signal', color='red', linewidth=1.5)
        
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in self.df['MACD_Hist']]
        ax3.bar(self.df.index, self.df['MACD_Hist'], label='Histogram', color=colors, alpha=0.3)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_ylabel('MACD', fontsize=10)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Volume and OBV
        ax4 = plt.subplot(gs[3], sharex=ax1)
        colors_vol = ['green' if self.df['Close'].iloc[i] >= self.df['Close'].iloc[i-1] else 'red' 
                      for i in range(len(self.df))]
        ax4.bar(self.df.index, self.df['Volume'], color=colors_vol, alpha=0.4, label='Volume')
        ax4.set_ylabel('Volume', fontsize=10, color='blue')
        ax4.tick_params(axis='y', labelcolor='blue')
        ax4.legend(loc='upper left', fontsize=8)
        
        # OBV on secondary axis
        ax4_2 = ax4.twinx()
        ax4_2.plot(self.df.index, self.df['OBV'], color='orange', linewidth=1.5, label='OBV')
        ax4_2.set_ylabel('OBV', fontsize=10, color='orange')
        ax4_2.tick_params(axis='y', labelcolor='orange')
        ax4_2.legend(loc='upper right', fontsize=8)
        
        # Panel 5: Stochastic
        ax5 = plt.subplot(gs[4], sharex=ax1)
        ax5.plot(self.df.index, self.df['Stochastic'], label='Stochastic', color='purple', linewidth=1.5)
        ax5.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Overbought (80)')
        ax5.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='Oversold (20)')
        ax5.fill_between(self.df.index, 80, 100, alpha=0.1, color='red')
        ax5.fill_between(self.df.index, 0, 20, alpha=0.1, color='green')
        ax5.set_ylabel('Stochastic', fontsize=10)
        ax5.set_xlabel('Date', fontsize=12)
        ax5.set_ylim(0, 100)
        ax5.legend(loc='upper left', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Add info text
        current_price = self.df['Close'].iloc[-1]
        rsi = self.df['RSI'].iloc[-1]
        macd = self.df['MACD'].iloc[-1]
        
        info_text = f"Current: ${current_price:.2f} | RSI: {rsi:.1f} | MACD: {macd:.2f}"
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def create_candlestick_chart(self, save_path=None):
        """Create candlestick chart with volume"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Candlestick data
        for idx in range(len(self.df)):
            date = self.df.index[idx]
            open_price = self.df['Open'].iloc[idx]
            close_price = self.df['Close'].iloc[idx]
            high = self.df['High'].iloc[idx]
            low = self.df['Low'].iloc[idx]
            
            color = 'green' if close_price >= open_price else 'red'
            
            # Draw high-low line
            ax1.plot([date, date], [low, high], color='black', linewidth=0.5)
            
            # Draw open-close rectangle
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)
            rect = Rectangle((date, bottom), width=0.6, height=height,
                           facecolor=color, edgecolor='black', linewidth=0.5)
            ax1.add_patch(rect)
        
        # Add MAs
        ax1.plot(self.df.index, self.df['MA20'], label='MA20', color='blue', linewidth=1.5, alpha=0.7)
        ax1.plot(self.df.index, self.df['MA50'], label='MA50', color='orange', linewidth=1.5, alpha=0.7)
        
        ax1.set_title(f'{self.ticker} - Candlestick Chart', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume
        colors_vol = ['green' if self.df['Close'].iloc[i] >= self.df['Open'].iloc[i] else 'red' 
                      for i in range(len(self.df))]
        ax2.bar(self.df.index, self.df['Volume'], color=colors_vol, alpha=0.5)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def create_indicator_dashboard(self, save_path=None):
        """Create dashboard showing all key indicators"""
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.3)
        
        current = self.df.iloc[-1]
        
        # 1. Price Performance
        ax1 = plt.subplot(gs[0, :2])
        returns = (self.df['Close'] / self.df['Close'].iloc[0] - 1) * 100
        ax1.plot(self.df.index, returns, linewidth=2, color='blue')
        ax1.fill_between(self.df.index, 0, returns, alpha=0.3, color='blue')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_title('Cumulative Return (%)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Current Stats
        ax2 = plt.subplot(gs[0, 2])
        ax2.axis('off')
        stats_text = f"""
        CURRENT STATS
        
        Price: ${current['Close']:.2f}
        
        MA20: ${current['MA20']:.2f}
        MA50: ${current['MA50']:.2f}
        
        RSI: {current['RSI']:.1f}
        MACD: {current['MACD']:.3f}
        
        ATR: ${current['ATR']:.2f}
        Volume: {current['Volume']:,.0f}
        """
        ax2.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 3. RSI Gauge
        ax3 = plt.subplot(gs[1, 0])
        rsi_val = current['RSI']
        ax3.barh(['RSI'], [rsi_val], color='red' if rsi_val > 70 else ('green' if rsi_val < 30 else 'yellow'))
        ax3.set_xlim(0, 100)
        ax3.axvline(x=70, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(x=30, color='green', linestyle='--', alpha=0.5)
        ax3.set_title(f'RSI: {rsi_val:.1f}', fontsize=10, fontweight='bold')
        
        # 4. MACD Trend
        ax4 = plt.subplot(gs[1, 1])
        recent_macd = self.df['MACD'].tail(30)
        recent_signal = self.df['MACD_Signal'].tail(30)
        ax4.plot(range(len(recent_macd)), recent_macd, label='MACD', color='blue')
        ax4.plot(range(len(recent_signal)), recent_signal, label='Signal', color='red')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('MACD (Last 30 Days)', fontsize=10, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Volatility (ATR)
        ax5 = plt.subplot(gs[1, 2])
        recent_atr = self.df['ATR'].tail(50)
        ax5.plot(range(len(recent_atr)), recent_atr, color='purple', linewidth=2)
        ax5.fill_between(range(len(recent_atr)), recent_atr, alpha=0.3, color='purple')
        ax5.set_title('ATR - Volatility', fontsize=10, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Volume Profile
        ax6 = plt.subplot(gs[2, 0])
        recent_vol = self.df['Volume'].tail(30)
        colors = ['green' if self.df['Close'].iloc[-(30-i)] >= self.df['Close'].iloc[-(31-i)] 
                 else 'red' for i in range(len(recent_vol))]
        ax6.bar(range(len(recent_vol)), recent_vol, color=colors, alpha=0.5)
        ax6.set_title('Volume (Last 30 Days)', fontsize=10, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. Trend Strength (ADX)
        ax7 = plt.subplot(gs[2, 1])
        recent_adx = self.df['ADX'].tail(50)
        ax7.plot(range(len(recent_adx)), recent_adx, color='orange', linewidth=2)
        ax7.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Strong Trend')
        ax7.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Weak Trend')
        ax7.fill_between(range(len(recent_adx)), 25, 100, alpha=0.1, color='red')
        ax7.set_title(f'ADX: {current["ADX"]:.1f} - Trend Strength', fontsize=10, fontweight='bold')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        # 8. Stochastic
        ax8 = plt.subplot(gs[2, 2])
        recent_stoch = self.df['Stochastic'].tail(30)
        ax8.plot(range(len(recent_stoch)), recent_stoch, color='purple', linewidth=2)
        ax8.axhline(y=80, color='red', linestyle='--', alpha=0.5)
        ax8.axhline(y=20, color='green', linestyle='--', alpha=0.5)
        ax8.fill_between(range(len(recent_stoch)), 80, 100, alpha=0.1, color='red')
        ax8.fill_between(range(len(recent_stoch)), 0, 20, alpha=0.1, color='green')
        ax8.set_title('Stochastic Oscillator', fontsize=10, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # Main title
        fig.suptitle(f'{self.ticker} - Technical Indicators Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")
        
        plt.show()
        
        return fig

# Example usage
if __name__ == "__main__":
    print("="*80)
    print("STOCK CHARTING SYSTEM")
    print("="*80)
    
    # Create charts for top biotech stocks
    tickers = ['NTLA', 'INSM', 'IONS']
    
    for ticker in tickers:
        print(f"\nGenerating charts for {ticker}...")
        
        try:
            charts = StockCharts(ticker, period='6mo')
            
            # Create comprehensive chart
            print(f"Creating comprehensive chart...")
            charts.create_comprehensive_chart(save_path=f'{ticker}_comprehensive.png')
            
            # Create dashboard
            print(f"Creating indicator dashboard...")
            charts.create_indicator_dashboard(save_path=f'{ticker}_dashboard.png')
            
            print(f"[OK] Charts created for {ticker}")
            
        except Exception as e:
            print(f"[ERROR] Failed to create charts for {ticker}: {str(e)[:100]}")
    
    print("\n" + "="*80)
    print("CHARTING COMPLETE")
    print("="*80)

