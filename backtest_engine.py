"""
Enhanced Backtesting Engine with Optimization
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from advanced_indicators import TechnicalIndicators

class BacktestEngine:
    """Enhanced backtesting engine with optimization and walk-forward testing"""
    
    def __init__(self, ticker, initial_capital=10000):
        self.ticker = ticker
        self.initial_capital = initial_capital
        self.df = None
        self.trades = []
        self.portfolio_value = []
        self.params = {
            'rsi_low': 40,
            'rsi_high': 60,
            'adx_threshold': 25,
            'stop_loss_pct': 8,
            'take_profit_pct': 20
        }
    
    def download_data(self, period='2y', split_ratio=None):
        """
        Download historical data
        
        Args:
            period: Time period to download
            split_ratio: If provided, splits data into train/test (e.g., 0.7 for 70/30 split)
        
        Returns:
            df or (train_df, test_df) if split_ratio provided
        """
        stock = yf.Ticker(self.ticker)
        df = stock.history(period=period, interval='1d')
        
        if df.empty:
            raise ValueError(f"No data found for {self.ticker}")
        
        # Add indicators
        df = self._calculate_indicators(df)
        
        if split_ratio:
            split_idx = int(len(df) * split_ratio)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            return train_df, test_df
        else:
            self.df = df
            return df
    
    def _calculate_indicators(self, df):
        """Calculate all required indicators"""
        # Moving Averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        df['RSI'] = TechnicalIndicators.RSI(df['Close'])
        
        # MACD
        macd_data = TechnicalIndicators.MACD(df['Close'])
        df['MACD'] = macd_data['MACD']
        df['MACD_Signal'] = macd_data['Signal']
        df['MACD_Hist'] = macd_data['Histogram']
        
        # ADX
        adx, plus_di, minus_di = TechnicalIndicators.ADX(df['High'], df['Low'], df['Close'])
        df['ADX'] = adx
        df['+DI'] = plus_di
        df['-DI'] = minus_di
        
        # Bollinger Bands
        bb = TechnicalIndicators.BollingerBands(df['Close'])
        df['BB_Upper'] = bb['Upper']
        df['BB_Middle'] = bb['Middle']
        df['BB_Lower'] = bb['Lower']
        
        # Volume
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        
        # ATR
        df['ATR'] = TechnicalIndicators.ATR(df['High'], df['Low'], df['Close'])
        
        return df.dropna()
    
    def set_parameters(self, **kwargs):
        """Set strategy parameters"""
        self.params.update(kwargs)
    
    def _generate_signals(self, df):
        """Generate buy/sell signals based on parameters"""
        params = self.params
        
        # BUY CONDITIONS
        df['Buy_Signal'] = (
            (df['Close'] > df['MA20']) &
            (df['MA20'] > df['MA50']) &
            (df['RSI'] > params['rsi_low']) &
            (df['RSI'] < params['rsi_high']) &
            (df['MACD'] > df['MACD_Signal']) &
            (df['ADX'] > params['adx_threshold']) &
            (df['Volume'] > df['Volume_MA20'])
        )
        
        # SELL CONDITIONS
        df['Sell_Signal'] = (
            (df['Close'] < df['MA20']) |
            (df['RSI'] > 75) |
            (df['MACD'] < df['MACD_Signal'])
        )
        
        return df
    
    def run_backtest(self, df=None):
        """
        Run backtest on data
        
        Args:
            df: DataFrame to backtest on (uses self.df if None)
        
        Returns:
            Dictionary with results
        """
        if df is None:
            df = self.df
        
        df = self._generate_signals(df)
        
        cash = self.initial_capital
        shares = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        trades = []
        portfolio_value = []
        equity_curve = []
        
        for idx, row in df.iterrows():
            date = idx
            price = row['Close']
            
            # Check if in position
            if shares > 0:
                # Check exit conditions
                exit_triggered = False
                exit_reason = ""
                
                # Stop loss
                if price <= stop_loss:
                    exit_triggered = True
                    exit_reason = "Stop Loss"
                
                # Take profit
                elif price >= take_profit:
                    exit_triggered = True
                    exit_reason = "Take Profit"
                
                # Technical sell signal
                elif row['Sell_Signal']:
                    exit_triggered = True
                    exit_reason = "Sell Signal"
                
                # Execute exit
                if exit_triggered:
                    cash = shares * price
                    pnl = ((price - entry_price) / entry_price) * 100
                    hold_days = (date - entry_date).days
                    
                    trades.append({
                        'Type': 'SELL',
                        'Date': date,
                        'Price': price,
                        'Shares': shares,
                        'Value': cash,
                        'PnL%': pnl,
                        'PnL$': cash - (shares * entry_price),
                        'HoldDays': hold_days,
                        'Reason': exit_reason
                    })
                    
                    shares = 0
                    entry_price = 0
            
            # Check buy conditions (only if not in position)
            if shares == 0 and row['Buy_Signal'] and cash > 0:
                shares = cash / price
                entry_price = price
                entry_date = date
                stop_loss = price * (1 - self.params['stop_loss_pct'] / 100)
                take_profit = price * (1 + self.params['take_profit_pct'] / 100)
                
                trades.append({
                    'Type': 'BUY',
                    'Date': date,
                    'Price': price,
                    'Shares': shares,
                    'Value': cash,
                    'PnL%': 0,
                    'PnL$': 0,
                    'HoldDays': 0,
                    'Reason': 'Entry Signal'
                })
                
                cash = 0
            
            # Track portfolio value and equity curve
            if shares > 0:
                current_value = shares * price
            else:
                current_value = cash
            
            portfolio_value.append({
                'Date': date,
                'Value': current_value,
                'Position': 'Long' if shares > 0 else 'Cash',
                'Price': price
            })
            
            equity_curve.append(current_value)
        
        # Close any open position at end
        if shares > 0:
            final_price = df['Close'].iloc[-1]
            cash = shares * final_price
            pnl = ((final_price - entry_price) / entry_price) * 100
            hold_days = (df.index[-1] - entry_date).days
            
            trades.append({
                'Type': 'SELL',
                'Date': df.index[-1],
                'Price': final_price,
                'Shares': shares,
                'Value': cash,
                'PnL%': pnl,
                'PnL$': cash - (shares * entry_price),
                'HoldDays': hold_days,
                'Reason': 'End of Period'
            })
        
        final_value = cash if shares == 0 else shares * df['Close'].iloc[-1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, portfolio_value, df, final_value)
        
        return {
            'trades': pd.DataFrame(trades),
            'portfolio_value': pd.DataFrame(portfolio_value),
            'equity_curve': equity_curve,
            'final_value': final_value,
            'metrics': metrics,
            'df': df
        }
    
    def _calculate_metrics(self, trades, portfolio_value, df, final_value):
        """Calculate performance metrics"""
        df_trades = pd.DataFrame(trades)
        df_portfolio = pd.DataFrame(portfolio_value)
        
        # Basic returns
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        buy_hold_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
        
        # Trade statistics
        if len(df_trades) > 0:
            sell_trades = df_trades[df_trades['Type'] == 'SELL']
            
            if len(sell_trades) > 0:
                winning_trades = sell_trades[sell_trades['PnL%'] > 0]
                losing_trades = sell_trades[sell_trades['PnL%'] <= 0]
                
                win_rate = (len(winning_trades) / len(sell_trades)) * 100
                avg_win = winning_trades['PnL%'].mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades['PnL%'].mean() if len(losing_trades) > 0 else 0
                avg_trade = sell_trades['PnL%'].mean()
                
                # Profit factor
                total_wins = winning_trades['PnL$'].sum() if len(winning_trades) > 0 else 0
                total_losses = abs(losing_trades['PnL$'].sum()) if len(losing_trades) > 0 else 1
                profit_factor = total_wins / total_losses if total_losses > 0 else 0
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                avg_trade = 0
                profit_factor = 0
            
            num_trades = len(df_trades[df_trades['Type'] == 'BUY'])
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            avg_trade = 0
            profit_factor = 0
            num_trades = 0
        
        # Sharpe Ratio (annualized)
        if len(df_portfolio) > 1:
            returns = df_portfolio['Value'].pct_change().dropna()
            if len(returns) > 0 and returns.std() != 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        if len(df_portfolio) > 0:
            equity = df_portfolio['Value'].values
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max * 100
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        # Calmar Ratio
        years = len(df) / 252
        cagr = ((final_value / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        calmar_ratio = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'outperformance': total_return - buy_hold_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'cagr': cagr
        }
    
    def optimize_parameters(self, param_grid, df=None):
        """
        Optimize strategy parameters
        
        Args:
            param_grid: Dictionary of parameters to test
                e.g., {'stop_loss_pct': [5, 8, 10], 'take_profit_pct': [15, 20, 25]}
            df: DataFrame to test on
        
        Returns:
            DataFrame with results for each parameter combination
        """
        if df is None:
            df = self.df
        
        results = []
        
        # Generate all combinations
        import itertools
        keys = param_grid.keys()
        values = param_grid.values()
        
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            
            # Set parameters
            original_params = self.params.copy()
            self.set_parameters(**params)
            
            # Run backtest
            result = self.run_backtest(df)
            
            # Store results
            result_row = {
                **params,
                **result['metrics']
            }
            results.append(result_row)
            
            # Restore original parameters
            self.params = original_params
        
        return pd.DataFrame(results)
    
    def walk_forward_test(self, train_ratio=0.7):
        """
        Walk-forward testing (train on first portion, test on last)
        
        Args:
            train_ratio: Ratio of data to use for training (e.g., 0.7 = 70% train, 30% test)
        
        Returns:
            Dictionary with train and test results
        """
        train_df, test_df = self.download_data(period='2y', split_ratio=train_ratio)
        
        # Run on training data
        train_result = self.run_backtest(train_df)
        
        # Run on test data
        test_result = self.run_backtest(test_df)
        
        return {
            'train': train_result,
            'test': test_result,
            'train_df': train_df,
            'test_df': test_df
        }

