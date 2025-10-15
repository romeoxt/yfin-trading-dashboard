"""
Complete Trading Strategy Implementation
Multi-indicator strategy with backtesting and live signals
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from advanced_indicators import TechnicalIndicators

class TradingStrategy:
    """
    Multi-Indicator Trading Strategy
    
    ENTRY RULES (ALL must be true):
    1. Price > MA20 > MA50 (trend confirmation)
    2. RSI between 40-60 (not overbought/oversold)
    3. MACD > Signal (momentum confirmation)
    4. ADX > 25 (strong trend)
    5. Volume > 20-day average (volume confirmation)
    
    EXIT RULES (ANY can trigger):
    1. Price crosses below MA20
    2. RSI > 75 (extreme overbought)
    3. MACD crosses below Signal
    4. Stop loss: 8% below entry
    5. Take profit: 20% above entry
    """
    
    def __init__(self, ticker, initial_capital=10000):
        self.ticker = ticker
        self.initial_capital = initial_capital
        self.df = None
        self.trades = []
        self.portfolio_value = []
    
    def download_data(self, period='2y'):
        """Download historical data"""
        print(f"Downloading {period} of data for {self.ticker}...")
        stock = yf.Ticker(self.ticker)
        self.df = stock.history(period=period, interval='1d')
        print(f"  Downloaded {len(self.df)} days of data")
    
    def calculate_indicators(self):
        """Calculate all required indicators"""
        print("Calculating indicators...")
        df = self.df
        
        # Moving Averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        df['RSI'] = TechnicalIndicators.RSI(df['Close'])
        
        # MACD
        macd, signal, hist = TechnicalIndicators.MACD(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        
        # ADX
        adx, plus_di, minus_di = TechnicalIndicators.ADX(df['High'], df['Low'], df['Close'])
        df['ADX'] = adx
        
        # Volume
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        
        # ATR for stop loss
        df['ATR'] = TechnicalIndicators.ATR(df['High'], df['Low'], df['Close'])
        
        self.df = df.dropna()
        print(f"  Indicators calculated")
    
    def generate_signals(self):
        """Generate buy/sell signals"""
        print("Generating trading signals...")
        df = self.df
        
        # BUY CONDITIONS
        df['Buy_Signal'] = (
            (df['Close'] > df['MA20']) &
            (df['MA20'] > df['MA50']) &
            (df['RSI'] > 40) &
            (df['RSI'] < 60) &
            (df['MACD'] > df['MACD_Signal']) &
            (df['ADX'] > 25) &
            (df['Volume'] > df['Volume_MA20'])
        )
        
        # SELL CONDITIONS
        df['Sell_Signal'] = (
            (df['Close'] < df['MA20']) |
            (df['RSI'] > 75) |
            (df['MACD'] < df['MACD_Signal'])
        )
        
        buy_count = df['Buy_Signal'].sum()
        sell_count = df['Sell_Signal'].sum()
        print(f"  Generated {buy_count} buy signals and {sell_count} sell signals")
        
        self.df = df
    
    def backtest(self):
        """Run backtest on historical data"""
        print("\nRunning backtest...")
        print("="*80)
        
        df = self.df
        cash = self.initial_capital
        shares = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        for idx, row in df.iterrows():
            date = idx
            price = row['Close']
            
            # Check if in position
            if shares > 0:
                # Check exit conditions
                exit_triggered = False
                exit_reason = ""
                
                # Stop loss (8%)
                if price <= stop_loss:
                    exit_triggered = True
                    exit_reason = "Stop Loss"
                
                # Take profit (20%)
                elif price >= take_profit:
                    exit_triggered = True
                    exit_reason = "Take Profit"
                
                # Technical sell signal
                elif row['Sell_Signal']:
                    exit_triggered = True
                    exit_reason = "Technical Sell Signal"
                
                # Execute exit
                if exit_triggered:
                    cash = shares * price
                    pnl = ((price - entry_price) / entry_price) * 100
                    
                    self.trades.append({
                        'Type': 'SELL',
                        'Date': date,
                        'Price': price,
                        'Shares': shares,
                        'Value': cash,
                        'PnL%': pnl,
                        'Reason': exit_reason
                    })
                    
                    shares = 0
                    entry_price = 0
            
            # Check buy conditions (only if not in position)
            if shares == 0 and row['Buy_Signal']:
                # Buy signal
                shares = cash / price
                entry_price = price
                stop_loss = price * 0.92  # 8% stop loss
                take_profit = price * 1.20  # 20% take profit
                
                self.trades.append({
                    'Type': 'BUY',
                    'Date': date,
                    'Price': price,
                    'Shares': shares,
                    'Value': cash,
                    'PnL%': 0,
                    'Reason': 'Entry Signal'
                })
                
                cash = 0
            
            # Track portfolio value
            if shares > 0:
                portfolio_val = shares * price
            else:
                portfolio_val = cash
            
            self.portfolio_value.append({
                'Date': date,
                'Value': portfolio_val,
                'Position': 'Long' if shares > 0 else 'Cash'
            })
        
        # Close any open position at end
        if shares > 0:
            final_price = df['Close'].iloc[-1]
            cash = shares * final_price
            pnl = ((final_price - entry_price) / entry_price) * 100
            
            self.trades.append({
                'Type': 'SELL',
                'Date': df.index[-1],
                'Price': final_price,
                'Shares': shares,
                'Value': cash,
                'PnL%': pnl,
                'Reason': 'End of Period'
            })
        
        self.final_value = cash if shares == 0 else shares * df['Close'].iloc[-1]
        self.df_trades = pd.DataFrame(self.trades)
        self.df_portfolio = pd.DataFrame(self.portfolio_value)
    
    def print_results(self):
        """Print backtest results"""
        print("\nBACKTEST RESULTS")
        print("="*80)
        
        # Overall performance
        total_return = ((self.final_value - self.initial_capital) / self.initial_capital) * 100
        
        # Buy and hold comparison
        buy_hold_return = ((self.df['Close'].iloc[-1] - self.df['Close'].iloc[0]) / 
                          self.df['Close'].iloc[0]) * 100
        
        print(f"\nPORTFOLIO PERFORMANCE:")
        print(f"  Initial Capital:    ${self.initial_capital:,.2f}")
        print(f"  Final Value:        ${self.final_value:,.2f}")
        print(f"  Total Return:       {total_return:+.2f}%")
        print(f"\n  Buy & Hold Return:  {buy_hold_return:+.2f}%")
        print(f"  Outperformance:     {(total_return - buy_hold_return):+.2f}%")
        
        # Trade statistics
        if len(self.df_trades) > 0:
            buy_trades = self.df_trades[self.df_trades['Type'] == 'BUY']
            sell_trades = self.df_trades[self.df_trades['Type'] == 'SELL']
            
            winning_trades = sell_trades[sell_trades['PnL%'] > 0]
            losing_trades = sell_trades[sell_trades['PnL%'] <= 0]
            
            win_rate = (len(winning_trades) / len(sell_trades) * 100) if len(sell_trades) > 0 else 0
            
            print(f"\nTRADE STATISTICS:")
            print(f"  Total Trades:       {len(buy_trades)}")
            print(f"  Winning Trades:     {len(winning_trades)}")
            print(f"  Losing Trades:      {len(losing_trades)}")
            print(f"  Win Rate:           {win_rate:.1f}%")
            
            if len(sell_trades) > 0:
                avg_win = winning_trades['PnL%'].mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades['PnL%'].mean() if len(losing_trades) > 0 else 0
                avg_trade = sell_trades['PnL%'].mean()
                
                print(f"  Average Trade:      {avg_trade:+.2f}%")
                print(f"  Average Win:        {avg_win:+.2f}%")
                print(f"  Average Loss:       {avg_loss:+.2f}%")
                
                # Exit reasons
                print(f"\nEXIT BREAKDOWN:")
                exit_reasons = sell_trades['Reason'].value_counts()
                for reason, count in exit_reasons.items():
                    print(f"  {reason:20s}: {count} trades")
        
        # Recent trades
        print(f"\nRECENT TRADES (Last 10):")
        print("-"*80)
        if len(self.df_trades) > 0:
            recent = self.df_trades.tail(10)
            for _, trade in recent.iterrows():
                date_str = trade['Date'].strftime('%Y-%m-%d')
                if trade['Type'] == 'BUY':
                    print(f"  {date_str}: {trade['Type']:4s} {trade['Shares']:.2f} shares @ ${trade['Price']:.2f} - {trade['Reason']}")
                else:
                    pnl_str = f"{trade['PnL%']:+.2f}%"
                    print(f"  {date_str}: {trade['Type']:4s} {trade['Shares']:.2f} shares @ ${trade['Price']:.2f} - {pnl_str:>8s} ({trade['Reason']})")
        
        print("="*80)
    
    def get_current_signal(self):
        """Get current trading signal"""
        print(f"\nCURRENT SIGNAL FOR {self.ticker}")
        print("="*80)
        
        latest = self.df.iloc[-1]
        price = latest['Close']
        
        print(f"\nCurrent Price: ${price:.2f}")
        print(f"Date: {self.df.index[-1].strftime('%Y-%m-%d')}")
        
        print(f"\nINDICATOR CHECK:")
        
        checks = {
            'Trend': (price > latest['MA20'] > latest['MA50'], 
                     f"Price ${price:.2f} > MA20 ${latest['MA20']:.2f} > MA50 ${latest['MA50']:.2f}"),
            'RSI': (40 < latest['RSI'] < 60, 
                   f"RSI {latest['RSI']:.1f} (target: 40-60)"),
            'MACD': (latest['MACD'] > latest['MACD_Signal'], 
                    f"MACD {latest['MACD']:.3f} > Signal {latest['MACD_Signal']:.3f}"),
            'Trend Strength': (latest['ADX'] > 25, 
                              f"ADX {latest['ADX']:.1f} (target: >25)"),
            'Volume': (latest['Volume'] > latest['Volume_MA20'], 
                      f"Volume {latest['Volume']:,.0f} > Avg {latest['Volume_MA20']:,.0f}")
        }
        
        passed = 0
        for name, (result, description) in checks.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {status} {name:15s}: {description}")
            if result:
                passed += 1
        
        print(f"\nSCORE: {passed}/5 conditions met")
        
        if latest['Buy_Signal']:
            print("\n>>> STRONG BUY SIGNAL! <<<")
            print(f"    Stop Loss: ${price * 0.92:.2f} (-8%)")
            print(f"    Take Profit: ${price * 1.20:.2f} (+20%)")
        elif latest['Sell_Signal']:
            print("\n>>> SELL SIGNAL <<<")
        else:
            print("\n>>> NEUTRAL / HOLD <<<")
        
        print("="*80)

# Example usage
if __name__ == "__main__":
    print("="*80)
    print("TRADING STRATEGY BACKTEST")
    print("="*80)
    
    # Test on multiple stocks
    test_tickers = ['NTLA', 'INSM', 'IONS']
    
    for ticker in test_tickers:
        print(f"\n\n{'#'*80}")
        print(f"TESTING: {ticker}")
        print(f"{'#'*80}")
        
        try:
            strategy = TradingStrategy(ticker, initial_capital=10000)
            
            # Run backtest
            strategy.download_data(period='2y')
            strategy.calculate_indicators()
            strategy.generate_signals()
            strategy.backtest()
            strategy.print_results()
            
            # Current signal
            strategy.get_current_signal()
            
        except Exception as e:
            print(f"Error testing {ticker}: {str(e)}")
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)

