"""
News Analyzer - Fetch and analyze stock news
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import re
from textblob import TextBlob

class NewsAnalyzer:
    """Fetch and analyze news for stocks"""
    
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
    
    def get_news(self, limit=20):
        """
        Fetch recent news for the stock
        
        Returns:
            DataFrame with news articles
        """
        try:
            news = self.stock.news
            
            if not news:
                return pd.DataFrame()
            
            news_data = []
            for article in news[:limit]:
                # Handle new Yahoo Finance API structure
                content = article.get('content', {})
                
                # Extract title from content
                title = content.get('title', 'No title')
                
                # Extract publisher from content.provider
                provider = content.get('provider', {})
                publisher = provider.get('displayName', 'Unknown')
                
                # Extract link from content.canonicalUrl or content.clickThroughUrl
                link = ''
                if 'canonicalUrl' in content and content['canonicalUrl']:
                    link = content['canonicalUrl'].get('url', '')
                elif 'clickThroughUrl' in content and content['clickThroughUrl']:
                    link = content['clickThroughUrl'].get('url', '')
                
                # Extract published date
                pub_date_str = content.get('pubDate', '')
                published = datetime.now()  # Default to now
                if pub_date_str:
                    try:
                        # Parse ISO format date
                        from dateutil import parser
                        published = parser.parse(pub_date_str)
                        # Make timezone-naive for consistency
                        if published.tzinfo is not None:
                            published = published.replace(tzinfo=None)
                    except:
                        pass
                
                # Extract thumbnail
                thumbnail = ''
                if 'thumbnail' in content and content['thumbnail']:
                    resolutions = content['thumbnail'].get('resolutions', [])
                    if resolutions:
                        thumbnail = resolutions[0].get('url', '')
                
                news_data.append({
                    'title': title,
                    'publisher': publisher,
                    'link': link,
                    'published': published,
                    'type': content.get('contentType', 'STORY'),
                    'thumbnail': thumbnail,
                    'summary': content.get('summary', ''),
                    'description': content.get('description', '')
                })
            
            return pd.DataFrame(news_data)
        
        except Exception as e:
            print(f"Error fetching news: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob
        
        Returns:
            dict with sentiment scores
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment = 'BULLISH'
                emoji = '+'
            elif polarity < -0.1:
                sentiment = 'BEARISH'
                emoji = '-'
            else:
                sentiment = 'NEUTRAL'
                emoji = '='
            
            return {
                'sentiment': sentiment,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'emoji': emoji,
                'confidence': abs(polarity)
            }
        
        except:
            return {
                'sentiment': 'NEUTRAL',
                'polarity': 0,
                'subjectivity': 0.5,
                'emoji': '=',
                'confidence': 0
            }
    
    def get_news_with_sentiment(self, limit=20):
        """
        Get news articles with sentiment analysis
        
        Returns:
            DataFrame with news and sentiment
        """
        df = self.get_news(limit)
        
        if df.empty:
            return df
        
        # Analyze sentiment for each article
        sentiments = []
        for title in df['title']:
            sent = self.analyze_sentiment(title)
            sentiments.append(sent)
        
        # Add sentiment columns
        df['sentiment'] = [s['sentiment'] for s in sentiments]
        df['polarity'] = [s['polarity'] for s in sentiments]
        df['confidence'] = [s['confidence'] for s in sentiments]
        
        # Calculate time ago
        now = datetime.now()
        df['time_ago'] = df['published'].apply(lambda x: self._format_time_ago(now - x))
        
        return df
    
    def _format_time_ago(self, delta):
        """Format timedelta as human readable string"""
        seconds = int(delta.total_seconds())
        
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes}m ago"
        elif seconds < 86400:
            hours = seconds // 3600
            return f"{hours}h ago"
        elif seconds < 604800:
            days = seconds // 86400
            return f"{days}d ago"
        else:
            weeks = seconds // 604800
            return f"{weeks}w ago"
    
    def get_news_summary(self, limit=20):
        """
        Get summary statistics of recent news sentiment
        
        Returns:
            dict with summary stats
        """
        df = self.get_news_with_sentiment(limit)
        
        if df.empty:
            return {
                'total_articles': 0,
                'bullish': 0,
                'bearish': 0,
                'neutral': 0,
                'avg_sentiment': 0,
                'sentiment_trend': 'NEUTRAL'
            }
        
        bullish = len(df[df['sentiment'] == 'BULLISH'])
        bearish = len(df[df['sentiment'] == 'BEARISH'])
        neutral = len(df[df['sentiment'] == 'NEUTRAL'])
        
        avg_sentiment = df['polarity'].mean()
        
        # Determine trend
        if avg_sentiment > 0.1:
            trend = 'BULLISH'
        elif avg_sentiment < -0.1:
            trend = 'BEARISH'
        else:
            trend = 'NEUTRAL'
        
        return {
            'total_articles': len(df),
            'bullish': bullish,
            'bearish': bearish,
            'neutral': neutral,
            'bullish_pct': (bullish / len(df)) * 100,
            'bearish_pct': (bearish / len(df)) * 100,
            'avg_sentiment': avg_sentiment,
            'sentiment_trend': trend
        }
    
    def detect_keywords(self, df, keywords):
        """
        Detect important keywords in news (e.g., 'FDA', 'approval', 'trial')
        
        Args:
            df: News DataFrame
            keywords: List of keywords to search for
        
        Returns:
            DataFrame filtered by keywords
        """
        if df.empty:
            return df
        
        # Search in titles (case-insensitive)
        mask = df['title'].str.contains('|'.join(keywords), case=False, na=False)
        return df[mask]
    
    def get_biotech_news(self, limit=20):
        """
        Get news with biotech-specific keyword detection
        
        Returns:
            dict with categorized news
        """
        df = self.get_news_with_sentiment(limit)
        
        if df.empty:
            return {'all': df, 'fda': df, 'trial': df, 'earnings': df}
        
        # Biotech-specific keywords
        fda_keywords = ['FDA', 'approval', 'approved', 'clearance', 'regulatory']
        trial_keywords = ['trial', 'study', 'phase', 'clinical', 'data', 'results']
        earnings_keywords = ['earnings', 'revenue', 'profit', 'quarter', 'Q1', 'Q2', 'Q3', 'Q4']
        
        return {
            'all': df,
            'fda': self.detect_keywords(df, fda_keywords),
            'trial': self.detect_keywords(df, trial_keywords),
            'earnings': self.detect_keywords(df, earnings_keywords)
        }


# Example usage
if __name__ == "__main__":
    # Test with biotech stocks
    tickers = ['NTLA', 'CRSP', 'EDIT']
    
    for ticker in tickers:
        print(f"\n{'='*80}")
        print(f"NEWS ANALYSIS: {ticker}")
        print(f"{'='*80}")
        
        analyzer = NewsAnalyzer(ticker)
        
        # Get summary
        summary = analyzer.get_news_summary(limit=10)
        print(f"\nSentiment Summary:")
        print(f"  Total Articles: {summary['total_articles']}")
        print(f"  Bullish: {summary['bullish']} ({summary['bullish_pct']:.1f}%)")
        print(f"  Bearish: {summary['bearish']} ({summary['bearish_pct']:.1f}%)")
        print(f"  Neutral: {summary['neutral']}")
        print(f"  Overall Trend: {summary['sentiment_trend']}")
        print(f"  Avg Sentiment: {summary['avg_sentiment']:.3f}")
        
        # Get recent news
        news_df = analyzer.get_news_with_sentiment(limit=5)
        
        if not news_df.empty:
            print(f"\nRecent Headlines:")
            for idx, row in news_df.iterrows():
                print(f"\n  [{row['time_ago']}] {row['sentiment']}")
                print(f"  {row['title']}")
                print(f"  Source: {row['publisher']}")

