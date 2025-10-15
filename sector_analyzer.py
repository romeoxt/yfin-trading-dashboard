"""
Sector Analyzer - Compare stocks to their sector/industry peers
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SectorAnalyzer:
    """Analyze stock performance relative to sector and peers"""
    
    # Predefined peer groups for common biotech stocks
    PEER_GROUPS = {
        # CRISPR Gene Editing
        'NTLA': ['CRSP', 'EDIT', 'BEAM', 'VERV', 'BLUE'],
        'CRSP': ['NTLA', 'EDIT', 'BEAM', 'VERV', 'BLUE'],
        'EDIT': ['NTLA', 'CRSP', 'BEAM', 'VERV', 'BLUE'],
        
        # RNA Therapeutics
        'IONS': ['AKCA', 'ARWR', 'DRNA', 'ALNY'],
        'MRNA': ['BNTX', 'NVAX', 'CVAC'],
        
        # Oncology
        'INSM': ['SGEN', 'IMMU', 'LEGN', 'EXEL'],
        
        # Rare Disease
        'RARE': ['FOLD', 'SRPT', 'BMRN', 'XLRN'],
        
        # General Biotech
        'GILD': ['VRTX', 'REGN', 'BIIB', 'AMGN'],
        'VRTX': ['GILD', 'REGN', 'BIIB', 'AMGN'],
    }
    
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.info = None
        try:
            self.info = self.stock.info
        except:
            pass
    
    def get_peers(self):
        """
        Get peer companies for the stock
        
        Returns:
            List of peer tickers
        """
        # Check predefined groups first
        if self.ticker in self.PEER_GROUPS:
            return self.PEER_GROUPS[self.ticker]
        
        # Fallback: use sector/industry from stock info
        if self.info:
            # This is a placeholder - in production you'd query a database
            sector = self.info.get('sector', '')
            if 'Healthcare' in sector or 'Biotechnology' in sector:
                # Return some general biotech peers
                return ['GILD', 'VRTX', 'REGN', 'BIIB', 'AMGN']
        
        return []
    
    def compare_performance(self, period='1y'):
        """
        Compare stock performance vs peers
        
        Args:
            period: Time period (1mo, 3mo, 6mo, 1y, 2y)
        
        Returns:
            DataFrame with comparison data
        """
        peers = self.get_peers()
        
        if not peers:
            return pd.DataFrame()
        
        # Add self to comparison
        all_tickers = [self.ticker] + peers
        
        comparison_data = []
        
        for tick in all_tickers:
            try:
                stock = yf.Ticker(tick)
                hist = stock.history(period=period)
                
                if hist.empty:
                    continue
                
                # Calculate metrics
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                total_return = ((end_price - start_price) / start_price) * 100
                
                # Volatility (annualized std dev)
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                
                # Max drawdown
                cum_returns = (1 + returns).cumprod()
                running_max = cum_returns.expanding().max()
                drawdown = (cum_returns - running_max) / running_max
                max_drawdown = drawdown.min() * 100
                
                # Sharpe ratio (assuming 0% risk-free rate)
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                
                # Get company info
                info = stock.info
                market_cap = info.get('marketCap', 0)
                
                comparison_data.append({
                    'Ticker': tick,
                    'Return': total_return,
                    'Volatility': volatility,
                    'Max Drawdown': max_drawdown,
                    'Sharpe Ratio': sharpe,
                    'Market Cap': market_cap,
                    'Is Target': tick == self.ticker
                })
            
            except Exception as e:
                print(f"Error fetching {tick}: {e}")
                continue
        
        df = pd.DataFrame(comparison_data)
        
        if not df.empty:
            # Calculate rankings
            df['Return Rank'] = df['Return'].rank(ascending=False)
            df['Sharpe Rank'] = df['Sharpe Ratio'].rank(ascending=False)
            
            # Calculate relative strength (percentile)
            target_return = df[df['Is Target']]['Return'].values[0] if any(df['Is Target']) else 0
            df['Relative Strength'] = df['Return'].apply(lambda x: 
                (df['Return'] < x).sum() / len(df) * 100
            )
        
        return df.sort_values('Return', ascending=False)
    
    def get_sector_summary(self, period='1y'):
        """
        Get summary stats for sector comparison
        
        Returns:
            dict with summary statistics
        """
        df = self.compare_performance(period)
        
        if df.empty:
            return {}
        
        target = df[df['Is Target']]
        
        if target.empty:
            return {}
        
        target = target.iloc[0]
        
        # Peer stats
        peers = df[~df['Is Target']]
        
        return {
            'ticker': self.ticker,
            'return': target['Return'],
            'rank': int(target['Return Rank']),
            'total_stocks': len(df),
            'relative_strength': target['Relative Strength'],
            'peer_avg_return': peers['Return'].mean() if not peers.empty else 0,
            'peer_median_return': peers['Return'].median() if not peers.empty else 0,
            'outperformance': target['Return'] - peers['Return'].mean() if not peers.empty else 0,
            'volatility': target['Volatility'],
            'sharpe_ratio': target['Sharpe Ratio'],
            'sharpe_rank': int(target['Sharpe Rank']),
            'market_cap': target['Market Cap'],
            'period': period
        }
    
    def get_correlation_matrix(self, period='1y'):
        """
        Calculate correlation matrix between stock and peers
        
        Returns:
            DataFrame with correlation matrix
        """
        peers = self.get_peers()
        
        if not peers:
            return pd.DataFrame()
        
        all_tickers = [self.ticker] + peers
        
        # Download price data
        data = yf.download(all_tickers, period=period, progress=False)['Close']
        
        if data.empty:
            return pd.DataFrame()
        
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # Calculate correlation
        corr_matrix = returns.corr()
        
        return corr_matrix
    
    def get_relative_strength_score(self, period='6mo'):
        """
        Calculate relative strength score (0-100)
        
        100 = outperforming all peers
        0 = underperforming all peers
        
        Returns:
            float: Relative strength score
        """
        df = self.compare_performance(period)
        
        if df.empty:
            return 50  # Neutral
        
        target = df[df['Is Target']]
        
        if target.empty:
            return 50
        
        return target.iloc[0]['Relative Strength']
    
    def get_sector_info(self):
        """Get sector and industry information"""
        if not self.info:
            return {
                'sector': 'Unknown',
                'industry': 'Unknown',
                'peers_count': len(self.get_peers())
            }
        
        return {
            'sector': self.info.get('sector', 'Unknown'),
            'industry': self.info.get('industry', 'Unknown'),
            'peers_count': len(self.get_peers()),
            'peers': self.get_peers()
        }


# Example usage
if __name__ == "__main__":
    # Test with CRISPR stocks
    tickers = ['NTLA', 'INSM', 'IONS']
    
    for ticker in tickers:
        print(f"\n{'='*80}")
        print(f"SECTOR ANALYSIS: {ticker}")
        print(f"{'='*80}")
        
        analyzer = SectorAnalyzer(ticker)
        
        # Get sector info
        sector_info = analyzer.get_sector_info()
        print(f"\nSector Info:")
        print(f"  Sector: {sector_info['sector']}")
        print(f"  Industry: {sector_info['industry']}")
        print(f"  Peers: {', '.join(sector_info['peers'])}")
        
        # Get performance comparison
        summary = analyzer.get_sector_summary(period='1y')
        
        if summary:
            print(f"\nPerformance (1Y):")
            print(f"  Return: {summary['return']:.2f}%")
            print(f"  Rank: #{summary['rank']} of {summary['total_stocks']}")
            print(f"  Relative Strength: {summary['relative_strength']:.1f}/100")
            print(f"  Peer Avg: {summary['peer_avg_return']:.2f}%")
            print(f"  Outperformance: {summary['outperformance']:+.2f}%")
            print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        
        # Show top performers
        df = analyzer.compare_performance(period='1y')
        
        if not df.empty:
            print(f"\nPeer Comparison:")
            for idx, row in df.head(5).iterrows():
                is_target = " (YOU)" if row['Is Target'] else ""
                print(f"  {row['Ticker']}{is_target}: {row['Return']:+.2f}% | Sharpe: {row['Sharpe Ratio']:.2f}")

