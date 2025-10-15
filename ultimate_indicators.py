"""
Ultimate Technical Indicators Package
Includes: Ichimoku Cloud, Fibonacci Retracements, and all previous indicators
"""

import pandas as pd
import numpy as np

class AdvancedIndicators:
    """Advanced technical indicators beyond basics"""
    
    @staticmethod
    def IchimokuCloud(high, low, close):
        """
        Ichimoku Cloud - Japanese multi-indicator system
        
        Returns:
        - Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        - Kijun-sen (Base Line): (26-period high + 26-period low)/2
        - Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, shifted 26 periods ahead
        - Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, shifted 26 periods ahead
        - Chikou Span (Lagging Span): Close shifted 26 periods back
        
        Signals:
        - Price above cloud = Bullish
        - Price below cloud = Bearish
        - Price in cloud = Neutral/consolidation
        - Tenkan crosses above Kijun = Buy signal
        - Tenkan crosses below Kijun = Sell signal
        """
        # Tenkan-sen (Conversion Line): 9-period
        period9_high = high.rolling(window=9).max()
        period9_low = low.rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): 26-period
        period26_high = high.rolling(window=26).max()
        period26_low = low.rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): Shifted 26 periods ahead
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): 52-period, shifted 26 periods ahead
        period52_high = high.rolling(window=52).max()
        period52_low = low.rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close shifted 26 periods back
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def FibonacciRetracement(high, low, lookback=30):
        """
        Fibonacci Retracement Levels
        
        Calculates key retracement levels based on recent swing high/low
        
        Levels:
        - 0.0% (High)
        - 23.6% retracement
        - 38.2% retracement
        - 50.0% retracement (not true Fib, but widely used)
        - 61.8% retracement (Golden Ratio)
        - 78.6% retracement
        - 100.0% (Low)
        
        Usage:
        - In uptrend: Buy at 38.2%, 50%, or 61.8% retracement
        - In downtrend: Sell at 38.2%, 50%, or 61.8% bounce
        """
        # Find swing high and low in lookback period
        recent_high = high.rolling(window=lookback).max()
        recent_low = low.rolling(window=lookback).min()
        
        diff = recent_high - recent_low
        
        # Calculate Fibonacci levels
        levels = {
            'level_0': recent_high,  # 0% (top)
            'level_236': recent_high - (diff * 0.236),
            'level_382': recent_high - (diff * 0.382),
            'level_500': recent_high - (diff * 0.500),
            'level_618': recent_high - (diff * 0.618),  # Golden Ratio
            'level_786': recent_high - (diff * 0.786),
            'level_100': recent_low  # 100% (bottom)
        }
        
        return levels
    
    @staticmethod
    def VolumeWeightedAveragePrice(high, low, close, volume):
        """
        VWAP - Volume Weighted Average Price
        
        The average price weighted by volume
        - Price above VWAP = Bullish
        - Price below VWAP = Bearish
        - Institutions use VWAP as execution benchmark
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def ParabolicSAR(high, low, close, af=0.02, max_af=0.2):
        """
        Parabolic SAR (Stop and Reverse)
        
        Trailing stop indicator
        - SAR below price = Uptrend, use as stop-loss
        - SAR above price = Downtrend, use as resistance
        - When price crosses SAR = Trend reversal signal
        """
        sar = close.copy()
        ep = high.copy()
        trend = pd.Series([1] * len(close), index=close.index)  # 1 = up, -1 = down
        acceleration = af
        
        for i in range(1, len(close)):
            prev_sar = sar.iloc[i-1]
            prev_ep = ep.iloc[i-1]
            
            if trend.iloc[i-1] == 1:  # Uptrend
                sar.iloc[i] = prev_sar + acceleration * (prev_ep - prev_sar)
                
                if low.iloc[i] < sar.iloc[i]:  # Trend reversal
                    trend.iloc[i] = -1
                    sar.iloc[i] = prev_ep
                    ep.iloc[i] = low.iloc[i]
                    acceleration = af
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > prev_ep:
                        ep.iloc[i] = high.iloc[i]
                        acceleration = min(acceleration + af, max_af)
                    else:
                        ep.iloc[i] = prev_ep
            else:  # Downtrend
                sar.iloc[i] = prev_sar - acceleration * (prev_sar - prev_ep)
                
                if high.iloc[i] > sar.iloc[i]:  # Trend reversal
                    trend.iloc[i] = 1
                    sar.iloc[i] = prev_ep
                    ep.iloc[i] = high.iloc[i]
                    acceleration = af
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < prev_ep:
                        ep.iloc[i] = low.iloc[i]
                        acceleration = min(acceleration + af, max_af)
                    else:
                        ep.iloc[i] = prev_ep
        
        return sar, trend
    
    @staticmethod
    def SuperTrend(high, low, close, period=10, multiplier=3):
        """
        SuperTrend Indicator
        
        Combination of ATR and price action
        - Green/Above = Uptrend
        - Red/Below = Downtrend
        - Very popular for trend following
        """
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate basic upper and lower bands
        hl_avg = (high + low) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        # Calculate SuperTrend
        supertrend = pd.Series([0.0] * len(close), index=close.index)
        direction = pd.Series([1] * len(close), index=close.index)
        
        for i in range(1, len(close)):
            # Adjust bands
            if close.iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
            
            # Set SuperTrend value
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
        
        return supertrend, direction
    
    @staticmethod
    def ElderRayIndex(high, low, close, period=13):
        """
        Elder Ray Index (Bull Power & Bear Power)
        
        Measures buying and selling pressure
        - Bull Power = High - EMA
        - Bear Power = Low - EMA
        
        Signals:
        - Bull Power > 0 and rising = Strong bulls
        - Bear Power < 0 and falling = Strong bears
        """
        ema = close.ewm(span=period, adjust=False).mean()
        bull_power = high - ema
        bear_power = low - ema
        
        return bull_power, bear_power, ema
    
    @staticmethod
    def WilliamsR(high, low, close, period=14):
        """
        Williams %R
        
        Similar to Stochastic but inverted
        - 0 to -20 = Overbought
        - -80 to -100 = Oversold
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
        
        return williams_r
    
    @staticmethod
    def CommodityChannelIndex(high, low, close, period=20):
        """
        CCI - Commodity Channel Index
        
        Measures deviation from average price
        - Above +100 = Overbought
        - Below -100 = Oversold
        - Used for divergence signals
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )
        
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        
        return cci

# Export all from previous script
from advanced_indicators import TechnicalIndicators

__all__ = ['AdvancedIndicators', 'TechnicalIndicators']

