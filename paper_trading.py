"""
Paper Trading Engine - Simulate trading with virtual money
"""

import json
import os
from datetime import datetime
import yfinance as yf
import pandas as pd

PAPER_PORTFOLIO_FILE = 'paper_portfolio.json'
PAPER_TRADES_FILE = 'paper_trades.json'

class PaperTradingEngine:
    """Paper trading engine for risk-free practice"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.portfolio = self._load_portfolio()
        self.trades = self._load_trades()
    
    def _load_portfolio(self):
        """Load paper trading portfolio"""
        if os.path.exists(PAPER_PORTFOLIO_FILE):
            with open(PAPER_PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        return {
            'cash': self.initial_capital,
            'positions': {},
            'created_at': datetime.now().isoformat(),
            'initial_capital': self.initial_capital
        }
    
    def _save_portfolio(self):
        """Save paper trading portfolio"""
        with open(PAPER_PORTFOLIO_FILE, 'w') as f:
            json.dump(self.portfolio, f, indent=2)
    
    def _load_trades(self):
        """Load trade history"""
        if os.path.exists(PAPER_TRADES_FILE):
            with open(PAPER_TRADES_FILE, 'r') as f:
                return json.load(f)
        return []
    
    def _save_trades(self):
        """Save trade history"""
        with open(PAPER_TRADES_FILE, 'w') as f:
            json.dump(self.trades, f, indent=2)
    
    def get_current_price(self, ticker):
        """Get current price for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period='1d')
            if not data.empty:
                return data['Close'].iloc[-1]
            return None
        except:
            return None
    
    def buy(self, ticker, shares, limit_price=None):
        """
        Execute a buy order
        
        Args:
            ticker: Stock symbol
            shares: Number of shares
            limit_price: Optional limit price (uses market if None)
        
        Returns:
            dict: Trade result
        """
        ticker = ticker.upper()
        
        # Get current price
        if limit_price:
            price = limit_price
        else:
            price = self.get_current_price(ticker)
            if not price:
                return {'success': False, 'message': f'Could not get price for {ticker}'}
        
        # Calculate cost
        cost = shares * price
        commission = 0  # No commission in paper trading
        total_cost = cost + commission
        
        # Check if enough cash
        if total_cost > self.portfolio['cash']:
            return {
                'success': False,
                'message': f'Insufficient funds. Need ${total_cost:.2f}, have ${self.portfolio["cash"]:.2f}'
            }
        
        # Execute trade
        self.portfolio['cash'] -= total_cost
        
        # Add to positions
        if ticker in self.portfolio['positions']:
            # Average up
            pos = self.portfolio['positions'][ticker]
            total_shares = pos['shares'] + shares
            total_cost_basis = (pos['avg_price'] * pos['shares']) + (price * shares)
            new_avg_price = total_cost_basis / total_shares
            
            self.portfolio['positions'][ticker] = {
                'shares': total_shares,
                'avg_price': new_avg_price,
                'first_bought': pos['first_bought']
            }
        else:
            self.portfolio['positions'][ticker] = {
                'shares': shares,
                'avg_price': price,
                'first_bought': datetime.now().isoformat()
            }
        
        # Record trade
        trade = {
            'id': len(self.trades) + 1,
            'type': 'BUY',
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'total': total_cost,
            'timestamp': datetime.now().isoformat()
        }
        self.trades.append(trade)
        
        # Save
        self._save_portfolio()
        self._save_trades()
        
        return {
            'success': True,
            'message': f'Bought {shares} shares of {ticker} at ${price:.2f}',
            'trade': trade
        }
    
    def sell(self, ticker, shares, limit_price=None):
        """
        Execute a sell order
        
        Args:
            ticker: Stock symbol
            shares: Number of shares
            limit_price: Optional limit price (uses market if None)
        
        Returns:
            dict: Trade result
        """
        ticker = ticker.upper()
        
        # Check if we own this stock
        if ticker not in self.portfolio['positions']:
            return {'success': False, 'message': f'You do not own {ticker}'}
        
        pos = self.portfolio['positions'][ticker]
        
        if shares > pos['shares']:
            return {
                'success': False,
                'message': f'Cannot sell {shares} shares. You only own {pos["shares"]} shares.'
            }
        
        # Get current price
        if limit_price:
            price = limit_price
        else:
            price = self.get_current_price(ticker)
            if not price:
                return {'success': False, 'message': f'Could not get price for {ticker}'}
        
        # Calculate proceeds
        proceeds = shares * price
        commission = 0
        total_proceeds = proceeds - commission
        
        # Calculate P&L
        cost_basis = shares * pos['avg_price']
        pnl = proceeds - cost_basis
        pnl_pct = (pnl / cost_basis) * 100
        
        # Execute trade
        self.portfolio['cash'] += total_proceeds
        
        # Update position
        if shares == pos['shares']:
            # Sold entire position
            del self.portfolio['positions'][ticker]
        else:
            # Partial sell
            self.portfolio['positions'][ticker]['shares'] -= shares
        
        # Record trade
        trade = {
            'id': len(self.trades) + 1,
            'type': 'SELL',
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'total': total_proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'timestamp': datetime.now().isoformat()
        }
        self.trades.append(trade)
        
        # Save
        self._save_portfolio()
        self._save_trades()
        
        return {
            'success': True,
            'message': f'Sold {shares} shares of {ticker} at ${price:.2f} | P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)',
            'trade': trade,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        }
    
    def get_portfolio_value(self):
        """Calculate total portfolio value"""
        total_value = self.portfolio['cash']
        position_values = {}
        
        for ticker, pos in self.portfolio['positions'].items():
            current_price = self.get_current_price(ticker)
            if current_price:
                value = pos['shares'] * current_price
                total_value += value
                
                cost_basis = pos['shares'] * pos['avg_price']
                pnl = value - cost_basis
                pnl_pct = (pnl / cost_basis) * 100
                
                position_values[ticker] = {
                    'shares': pos['shares'],
                    'avg_price': pos['avg_price'],
                    'current_price': current_price,
                    'market_value': value,
                    'cost_basis': cost_basis,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                }
        
        total_pnl = total_value - self.portfolio['initial_capital']
        total_pnl_pct = (total_pnl / self.portfolio['initial_capital']) * 100
        
        return {
            'cash': self.portfolio['cash'],
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'positions': position_values,
            'num_positions': len(position_values)
        }
    
    def get_trade_history(self, limit=None):
        """Get trade history"""
        trades = self.trades.copy()
        trades.reverse()  # Most recent first
        
        if limit:
            trades = trades[:limit]
        
        return trades
    
    def get_performance_summary(self):
        """Get summary of trading performance"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']
        
        if not sell_trades:
            portfolio_value = self.get_portfolio_value()
            return {
                'total_trades': len([t for t in self.trades if t['type'] == 'BUY']),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': portfolio_value['total_pnl'],
                'avg_win': 0,
                'avg_loss': 0
            }
        
        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('pnl', 0) <= 0]
        
        total_pnl = sum(t.get('pnl', 0) for t in sell_trades)
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        return {
            'total_trades': len(sell_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(sell_trades)) * 100 if sell_trades else 0,
            'total_realized_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades and sum(t['pnl'] for t in losing_trades) != 0 else 0
        }
    
    def reset_portfolio(self, initial_capital=None):
        """Reset paper trading portfolio"""
        if initial_capital:
            self.initial_capital = initial_capital
        
        self.portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'created_at': datetime.now().isoformat(),
            'initial_capital': self.initial_capital
        }
        self.trades = []
        
        self._save_portfolio()
        self._save_trades()
        
        return {'success': True, 'message': f'Portfolio reset with ${initial_capital:,.2f}'}


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("PAPER TRADING DEMO")
    print("="*80)
    
    # Initialize engine
    engine = PaperTradingEngine(initial_capital=100000)
    
    # Buy some stocks
    print("\n1. Buying 100 shares of NTLA...")
    result = engine.buy('NTLA', 100)
    print(f"   {result['message']}")
    
    print("\n2. Buying 50 shares of CRSP...")
    result = engine.buy('CRSP', 50)
    print(f"   {result['message']}")
    
    # Check portfolio
    print("\n3. Portfolio Status:")
    portfolio = engine.get_portfolio_value()
    print(f"   Cash: ${portfolio['cash']:,.2f}")
    print(f"   Total Value: ${portfolio['total_value']:,.2f}")
    print(f"   P&L: ${portfolio['total_pnl']:,.2f} ({portfolio['total_pnl_pct']:+.2f}%)")
    
    print(f"\n   Positions:")
    for ticker, pos in portfolio['positions'].items():
        print(f"   - {ticker}: {pos['shares']} shares @ ${pos['avg_price']:.2f} | "
              f"Value: ${pos['market_value']:,.2f} | P&L: {pos['pnl_pct']:+.2f}%")

