"""
Enhanced Valuation Analysis
Implements DCF, P/S analysis, and multi-factor scoring like the article describes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

class EnhancedValuation:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.info = self.stock.info
        
    def get_dcf_analysis(self):
        """Discounted Cash Flow analysis"""
        try:
            # Get financial data
            financials = self.stock.financials
            cash_flow = self.stock.cashflow
            
            if financials.empty or cash_flow.empty:
                return None
                
            # Extract Free Cash Flow (if available)
            fcf_data = None
            if 'Free Cash Flow' in cash_flow.index:
                fcf_data = cash_flow.loc['Free Cash Flow']
            elif 'Operating Cash Flow' in cash_flow.index and 'Capital Expenditures' in cash_flow.index:
                fcf_data = cash_flow.loc['Operating Cash Flow'] - cash_flow.loc['Capital Expenditures']
            
            if fcf_data is None or fcf_data.empty:
                return None
                
            # Get current FCF (most recent)
            current_fcf = fcf_data.iloc[0] if not fcf_data.empty else 0
            
            # Simple DCF assumptions (would need more sophisticated modeling)
            growth_rate = 0.10  # 10% growth assumption
            discount_rate = 0.12  # 12% discount rate
            terminal_growth = 0.03  # 3% terminal growth
            
            # Project FCF for 5 years
            projected_fcf = []
            for year in range(1, 6):
                fcf = current_fcf * ((1 + growth_rate) ** year)
                discounted_fcf = fcf / ((1 + discount_rate) ** year)
                projected_fcf.append(discounted_fcf)
            
            # Terminal value
            terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            discounted_terminal = terminal_value / ((1 + discount_rate) ** 5)
            
            # Total DCF value
            total_dcf = sum(projected_fcf) + discounted_terminal
            
            # Get shares outstanding
            shares = self.info.get('sharesOutstanding', 1)
            dcf_per_share = total_dcf / shares if shares > 0 else 0
            
            # Current price
            current_price = self.info.get('currentPrice', 0)
            
            # Overvaluation percentage
            if current_price > 0:
                overvaluation = ((current_price - dcf_per_share) / dcf_per_share) * 100
            else:
                overvaluation = 0
                
            return {
                'dcf_per_share': dcf_per_share,
                'current_price': current_price,
                'overvaluation_pct': overvaluation,
                'current_fcf': current_fcf,
                'valuation_status': 'OVERVALUED' if overvaluation > 0 else 'UNDERVALUED'
            }
            
        except Exception as e:
            print(f"DCF analysis error: {e}")
            return None
    
    def get_ps_analysis(self):
        """Price-to-Sales ratio analysis"""
        try:
            current_price = self.info.get('currentPrice', 0)
            market_cap = self.info.get('marketCap', 0)
            revenue = self.info.get('totalRevenue', 0)
            
            if revenue <= 0:
                return None
                
            # Calculate P/S ratio
            ps_ratio = market_cap / revenue
            
            # Industry benchmarks (simplified)
            industry_avg = 10.1  # Biotech average from article
            peer_avg = 17.7      # Peer average from article
            
            # Determine if overvalued
            is_overvalued = ps_ratio > industry_avg
            
            return {
                'ps_ratio': ps_ratio,
                'industry_avg': industry_avg,
                'peer_avg': peer_avg,
                'is_overvalued': is_overvalued,
                'valuation_status': 'OVERVALUED' if is_overvalued else 'FAIR'
            }
            
        except Exception as e:
            print(f"P/S analysis error: {e}")
            return None
    
    def get_valuation_score(self):
        """Multi-factor valuation score (0-6 like in article)"""
        score = 0
        factors = []
        
        try:
            # Factor 1: P/E Ratio
            pe_ratio = self.info.get('trailingPE', 0)
            if pe_ratio > 0:
                if pe_ratio < 15:
                    score += 1
                    factors.append("✓ Low P/E ratio")
                else:
                    factors.append("✗ High P/E ratio")
            
            # Factor 2: P/S Ratio
            ps_analysis = self.get_ps_analysis()
            if ps_analysis and ps_analysis['ps_ratio'] < ps_analysis['industry_avg']:
                score += 1
                factors.append("✓ Low P/S ratio")
            else:
                factors.append("✗ High P/S ratio")
            
            # Factor 3: Debt-to-Equity
            debt_to_equity = self.info.get('debtToEquity', 0)
            if debt_to_equity < 0.5:
                score += 1
                factors.append("✓ Low debt")
            else:
                factors.append("✗ High debt")
            
            # Factor 4: Current Ratio
            current_ratio = self.info.get('currentRatio', 0)
            if current_ratio > 2:
                score += 1
                factors.append("✓ Strong liquidity")
            else:
                factors.append("✗ Weak liquidity")
            
            # Factor 5: Revenue Growth
            revenue_growth = self.info.get('revenueGrowth', 0)
            if revenue_growth and revenue_growth > 0.1:  # 10% growth
                score += 1
                factors.append("✓ Growing revenue")
            else:
                factors.append("✗ Declining revenue")
            
            # Factor 6: Profitability
            profit_margins = self.info.get('profitMargins', 0)
            if profit_margins and profit_margins > 0.1:  # 10% margin
                score += 1
                factors.append("✓ Profitable")
            else:
                factors.append("✗ Not profitable")
            
            return {
                'score': score,
                'max_score': 6,
                'factors': factors,
                'grade': 'A' if score >= 5 else ('B' if score >= 4 else ('C' if score >= 3 else 'D'))
            }
            
        except Exception as e:
            print(f"Valuation score error: {e}")
            return {'score': 0, 'max_score': 6, 'factors': [], 'grade': 'F'}
    
    def get_comprehensive_valuation(self):
        """Get all valuation metrics together"""
        dcf = self.get_dcf_analysis()
        ps = self.get_ps_analysis()
        score = self.get_valuation_score()
        
        return {
            'dcf': dcf,
            'ps_analysis': ps,
            'valuation_score': score,
            'overall_status': self._determine_overall_status(dcf, ps, score)
        }
    
    def _determine_overall_status(self, dcf, ps, score):
        """Determine overall valuation status"""
        if not dcf and not ps:
            return "INSUFFICIENT_DATA"
        
        overvalued_signals = 0
        total_signals = 0
        
        if dcf and dcf['overvaluation_pct'] > 0:
            overvalued_signals += 1
        if dcf:
            total_signals += 1
            
        if ps and ps['is_overvalued']:
            overvalued_signals += 1
        if ps:
            total_signals += 1
            
        if score['score'] < 3:  # Less than half the factors
            overvalued_signals += 1
        total_signals += 1
        
        if total_signals == 0:
            return "INSUFFICIENT_DATA"
        
        overvalued_ratio = overvalued_signals / total_signals
        
        if overvalued_ratio >= 0.67:
            return "OVERVALUED"
        elif overvalued_ratio <= 0.33:
            return "UNDERVALUED"
        else:
            return "FAIR_VALUE"
