"""
Portfolio Tracker - Track your actual holdings with real-time alerts
"""
import json
import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

PORTFOLIO_FILE = 'portfolio.json'

def load_portfolio():
    """Load portfolio from file"""
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    return {'positions': [], 'alerts': []}

def save_portfolio(portfolio):
    """Save portfolio to file"""
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2)

def add_position(ticker, position_type, quantity, entry_price, entry_date, strike=None, expiration=None, option_type=None):
    """
    Add a new position to portfolio
    
    Args:
        ticker: Stock symbol
        position_type: 'shares' or 'options'
        quantity: Number of shares/contracts
        entry_price: Price per share or premium per contract
        entry_date: Date of entry (YYYY-MM-DD)
        strike: Strike price for options
        expiration: Expiration date for options (YYYY-MM-DD)
        option_type: 'call' or 'put'
    """
    portfolio = load_portfolio()
    
    position = {
        'id': f"{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'ticker': ticker.upper(),
        'type': position_type,
        'quantity': quantity,
        'entry_price': entry_price,
        'entry_date': entry_date,
        'added_at': datetime.now().isoformat()
    }
    
    if position_type == 'options':
        position.update({
            'strike': strike,
            'expiration': expiration,
            'option_type': option_type
        })
    
    portfolio['positions'].append(position)
    save_portfolio(portfolio)
    return position

def remove_position(position_id):
    """Remove a position from portfolio"""
    portfolio = load_portfolio()
    portfolio['positions'] = [p for p in portfolio['positions'] if p['id'] != position_id]
    save_portfolio(portfolio)

def calculate_position_value(position, current_price):
    """Calculate current value and P&L for a position"""
    if position['type'] == 'shares':
        cost_basis = position['quantity'] * position['entry_price']
        current_value = position['quantity'] * current_price
        pnl = current_value - cost_basis
        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
        
        return {
            'cost_basis': cost_basis,
            'current_value': current_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        }
    
    elif position['type'] == 'options':
        # For options, entry_price is premium paid per contract
        # Current value would need options data (not easily available in free tier)
        cost_basis = position['quantity'] * position['entry_price'] * 100  # 100 shares per contract
        
        # Estimate intrinsic value for basic tracking
        strike = position['strike']
        if position['option_type'] == 'call':
            intrinsic = max(0, current_price - strike)
        else:  # put
            intrinsic = max(0, strike - current_price)
        
        estimated_value = position['quantity'] * intrinsic * 100
        pnl = estimated_value - cost_basis
        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
        
        return {
            'cost_basis': cost_basis,
            'current_value': estimated_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'intrinsic_value': intrinsic
        }

def check_alerts_for_position(position, current_data, technical_signal):
    """
    Check if position should trigger any alerts
    
    Returns list of alerts for this position
    """
    alerts = []
    ticker = position['ticker']
    current_price = current_data['Close']
    entry_price = position['entry_price']
    rsi = current_data['RSI']
    volume = current_data['Volume']
    volume_avg = current_data['Volume_MA20']
    
    # 1. Stop Loss Alert (5% below entry for shares, 50% for options)
    if position['type'] == 'shares':
        stop_loss_threshold = entry_price * 0.95
        if current_price < stop_loss_threshold:
            loss_pct = ((current_price - entry_price) / entry_price) * 100
            alerts.append({
                'type': 'STOP_LOSS',
                'severity': 'critical',
                'message': f"{ticker}: Price ${current_price:.2f} below stop loss (${stop_loss_threshold:.2f})",
                'detail': f"Down {loss_pct:.1f}% from entry",
                'action': 'Consider selling to limit losses'
            })
    else:  # options
        # Options lose value faster - 50% loss threshold
        current_val = calculate_position_value(position, current_price)
        if current_val['pnl_pct'] < -50:
            alerts.append({
                'type': 'STOP_LOSS',
                'severity': 'critical',
                'message': f"{ticker} option: Down {current_val['pnl_pct']:.1f}%",
                'detail': f"Current value: ${current_val['current_value']:.2f} vs Cost: ${current_val['cost_basis']:.2f}",
                'action': 'Option losing significant value'
            })
    
    # 2. Take Profit Alert (20% gain for shares, 100% for options)
    if position['type'] == 'shares':
        take_profit_threshold = entry_price * 1.20
        if current_price > take_profit_threshold:
            gain_pct = ((current_price - entry_price) / entry_price) * 100
            alerts.append({
                'type': 'TAKE_PROFIT',
                'severity': 'positive',
                'message': f"{ticker}: Up {gain_pct:.1f}% - Consider taking profits",
                'detail': f"Price ${current_price:.2f} vs Entry ${entry_price:.2f}",
                'action': 'Lock in gains or set trailing stop'
            })
    else:  # options
        current_val = calculate_position_value(position, current_price)
        if current_val['pnl_pct'] > 100:
            alerts.append({
                'type': 'TAKE_PROFIT',
                'severity': 'positive',
                'message': f"{ticker} option: Up {current_val['pnl_pct']:.1f}% - Take profits!",
                'detail': f"Doubled your money",
                'action': 'Consider selling to lock in gains'
            })
    
    # 3. Overbought Warning
    if rsi >= 75:
        alerts.append({
            'type': 'OVERBOUGHT',
            'severity': 'warning',
            'message': f"{ticker}: RSI extremely high ({rsi:.1f})",
            'detail': 'Stock may be due for pullback',
            'action': 'Watch for reversal or take partial profits'
        })
    
    # 4. Oversold Alert (potential to average down)
    if rsi <= 30 and position['type'] == 'shares':
        alerts.append({
            'type': 'OVERSOLD',
            'severity': 'info',
            'message': f"{ticker}: RSI very low ({rsi:.1f})",
            'detail': 'Stock may be oversold',
            'action': 'Consider averaging down if fundamentals strong'
        })
    
    # 5. Technical Signal Changed
    if technical_signal in ['SELL', 'HOLD / OVERBOUGHT']:
        alerts.append({
            'type': 'SIGNAL_CHANGE',
            'severity': 'warning',
            'message': f"{ticker}: Technical signal is {technical_signal}",
            'detail': 'Momentum may be turning negative',
            'action': 'Review position and consider reducing exposure'
        })
    
    # 6. Volume Spike (possible news or breakout)
    if volume > volume_avg * 2:
        alerts.append({
            'type': 'VOLUME_SPIKE',
            'severity': 'info',
            'message': f"{ticker}: Volume spike detected",
            'detail': f"Volume {volume:,.0f} vs avg {volume_avg:,.0f}",
            'action': 'Check for news or significant price movement'
        })
    
    # 7. Options Expiration Warning
    if position['type'] == 'options':
        exp_date = datetime.strptime(position['expiration'], '%Y-%m-%d')
        days_to_exp = (exp_date - datetime.now()).days
        
        if days_to_exp <= 7:
            alerts.append({
                'type': 'EXPIRATION',
                'severity': 'critical',
                'message': f"{ticker} option expires in {days_to_exp} days",
                'detail': f"Expiration: {position['expiration']}",
                'action': 'Close position or roll to later date'
            })
        elif days_to_exp <= 30:
            alerts.append({
                'type': 'EXPIRATION',
                'severity': 'warning',
                'message': f"{ticker} option expires in {days_to_exp} days",
                'detail': f"Time decay accelerating",
                'action': 'Monitor closely'
            })
    
    return alerts

def get_portfolio_summary():
    """Get full portfolio summary with all positions and alerts"""
    portfolio = load_portfolio()
    positions_data = []
    all_alerts = []
    total_cost = 0
    total_value = 0
    
    for position in portfolio['positions']:
        try:
            ticker = position['ticker']
            stock = yf.Ticker(ticker)
            df = stock.history(period='3mo')
            
            if df.empty:
                continue
            
            # Get current price and technical data
            latest = df.iloc[-1]
            current_price = latest['Close']
            
            # Calculate technical indicators
            from advanced_indicators import TechnicalIndicators
            df = TechnicalIndicators.add_all_indicators(df)
            latest = df.iloc[-1]
            
            # Get technical signal (simplified)
            from live_dashboard import get_trading_signal
            signal, color, score, conditions = get_trading_signal(df)
            
            # Calculate position value
            value_data = calculate_position_value(position, current_price)
            
            # Check for alerts
            position_alerts = check_alerts_for_position(position, latest, signal)
            
            position_data = {
                'position': position,
                'current_price': current_price,
                'signal': signal,
                'score': score,
                'rsi': latest['RSI'],
                'value_data': value_data,
                'alerts': position_alerts
            }
            
            positions_data.append(position_data)
            all_alerts.extend(position_alerts)
            
            total_cost += value_data['cost_basis']
            total_value += value_data['current_value']
            
        except Exception as e:
            print(f"Error processing {position['ticker']}: {e}")
            continue
    
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    
    return {
        'positions': positions_data,
        'alerts': all_alerts,
        'summary': {
            'total_cost': total_cost,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'num_positions': len(positions_data),
            'critical_alerts': len([a for a in all_alerts if a['severity'] == 'critical']),
            'warning_alerts': len([a for a in all_alerts if a['severity'] == 'warning'])
        }
    }

