"""
AI Stock Analyzer - Complete Server (Simplified Structure)
All files in the same directory
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from functools import lru_cache
import praw
from textblob import TextBlob
from newsapi import NewsApiClient
import finnhub
from dotenv import load_dotenv
from ml_predictor import train_and_predict_stock

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# API Keys
API_KEY = os.getenv('API_KEY', 'my-secret-stock-api-key-2024')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', 'demo')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'stock_analyzer_bot/1.0')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')

# API Endpoints
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


# ==========================================
# Technical Indicators
# ==========================================

def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ==========================================
# Sentiment Analysis
# ==========================================

def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        score = int(polarity * 100)
        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"
        return score, label
    except:
        return 0, "Neutral"


# ==========================================
# Stock Data Fetching
# ==========================================

def get_alpha_vantage_symbol(stock_code, market_type):
    if market_type == 'A-share':
        if stock_code.startswith('6'):
            return f"{stock_code}.SHH"
        else:
            return f"{stock_code}.SHZ"
    elif market_type == 'HK':
        return f"{stock_code}.HKG"
    else:
        return stock_code

def fetch_stock_data_for_ml(stock_code, market_type):
    try:
        symbol = get_alpha_vantage_symbol(stock_code, market_type)
        print(f"Fetching data for: {symbol}")
        
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': '5min',
            'outputsize': 'full',
            'apikey': ALPHA_VANTAGE_KEY
        }
        
        response = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=10)
        data = response.json()
        
        if 'Error Message' in data:
            return None, {'error': f'Invalid symbol: {stock_code}'}
        
        if 'Note' in data:
            return None, {'error': 'API rate limit reached'}
        
        json_key = 'Time Series (5min)'
        if json_key not in data:
            return None, {'error': 'No intraday data available'}
        
        time_series = data[json_key]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = df[col].astype(float)
        df['Volume'] = df['Volume'].astype(int)
        
        if df.empty:
            return None, {'error': f'No data found for {stock_code}'}
        
        # Calculate indicators
        df['SMA_5'] = calculate_sma(df['Close'], 5)
        df['SMA_10'] = calculate_sma(df['Close'], 10)
        df['SMA_20'] = calculate_sma(df['Close'], 20)
        df['SMA_60'] = calculate_sma(df['Close'], 60)
        
        macd, signal, histogram = calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = histogram
        df['RSI'] = calculate_rsi(df['Close'])
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        price_change = latest['Close'] - prev['Close']
        price_change_pct = (price_change / prev['Close']) * 100
        
        if latest['Close'] > latest['SMA_5'] > latest['SMA_10'] > latest['SMA_20']:
            trend = "Strong Uptrend"
        elif latest['Close'] > latest['SMA_10']:
            trend = "Uptrend"
        elif latest['Close'] < latest['SMA_10']:
            trend = "Downtrend"
        else:
            trend = "Sideways"
        
        stock_data = {
            'stock_code': stock_code,
            'market_type': market_type,
            'ticker': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': round(latest['Close'], 2),
            'price_change': round(price_change, 2),
            'price_change_percent': round(price_change_pct, 2),
            'volume': int(latest['Volume']),
            'high': round(latest['High'], 2),
            'low': round(latest['Low'], 2),
            'open': round(latest['Open'], 2),
            'trend': trend,
            'technical_indicators': {
                'SMA_5': round(latest['SMA_5'], 2) if pd.notna(latest['SMA_5']) else None,
                'SMA_10': round(latest['SMA_10'], 2) if pd.notna(latest['SMA_10']) else None,
                'SMA_20': round(latest['SMA_20'], 2) if pd.notna(latest['SMA_20']) else None,
                'SMA_60': round(latest['SMA_60'], 2) if pd.notna(latest['SMA_60']) else None,
                'MACD': round(latest['MACD'], 4) if pd.notna(latest['MACD']) else None,
                'MACD_Signal': round(latest['MACD_Signal'], 4) if pd.notna(latest['MACD_Signal']) else None,
                'MACD_Histogram': round(latest['MACD_Histogram'], 4) if pd.notna(latest['MACD_Histogram']) else None,
                'RSI': round(latest['RSI'], 2) if pd.notna(latest['RSI']) else None
            },
            'support_resistance': {
                'support_1': round(latest['Low'], 2),
                'support_2': round(df['Low'].tail(10).min(), 2),
                'resistance_1': round(latest['High'], 2),
                'resistance_2': round(df['High'].tail(10).max(), 2)
            },
            'volume_analysis': {
                'current_volume': int(latest['Volume']),
                'avg_volume_10d': int(df['Volume'].tail(10).mean()),
                'volume_ratio': round(latest['Volume'] / df['Volume'].tail(10).mean(), 2) if df['Volume'].tail(10).mean() > 0 else 0
            }
        }
        
        return df, stock_data
        
    except Exception as e:
        return None, {'error': str(e)}


# ==========================================
# Sentiment Sources
# ==========================================

def fetch_finnhub_sentiment(stock_code):
    if not FINNHUB_API_KEY:
        return {'error': 'Finnhub API key not configured'}
    
    try:
        print(f"📊 Fetching Finnhub data for {stock_code}...")
        finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
        
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        news = finnhub_client.company_news(stock_code, _from=from_date, to=to_date)
        
        if not news or len(news) == 0:
            return {
                'source': 'finnhub',
                'news_count': 0,
                'sentiment_score': 0,
                'sentiment_label': 'Neutral',
                'news_items': []
            }
        
        sentiments = []
        news_items = []
        
        for article in news[:15]:
            headline = article.get('headline', '')
            summary = article.get('summary', '')
            text = f"{headline}. {summary}"
            
            score, label = analyze_sentiment(text)
            sentiments.append(score)
            
            news_items.append({
                'headline': headline,
                'source': article.get('source', 'Unknown'),
                'sentiment': label,
                'datetime': datetime.fromtimestamp(article.get('datetime', 0)).strftime('%Y-%m-%d %H:%M')
            })
        
        avg_sentiment = int(np.mean(sentiments)) if sentiments else 0
        sentiment_label = "Positive" if avg_sentiment > 10 else "Negative" if avg_sentiment < -10 else "Neutral"
        
        print(f"✅ Finnhub: {sentiment_label} ({avg_sentiment}/100)")
        
        return {
            'source': 'finnhub',
            'news_count': len(news_items),
            'sentiment_score': avg_sentiment,
            'sentiment_label': sentiment_label,
            'news_items': news_items[:10],
            'confidence': 'high' if len(news_items) >= 5 else 'medium'
        }
        
    except Exception as e:
        print(f"❌ Finnhub error: {str(e)}")
        return {'error': str(e)}

def fetch_news_sentiment(stock_code):
    if not NEWS_API_KEY:
        return {'error': 'NewsAPI key not configured'}
    
    try:
        print(f"📰 Fetching news for {stock_code}...")
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        articles = newsapi.get_everything(
            q=stock_code,
            from_param=from_date,
            language='en',
            sort_by='relevancy',
            page_size=20
        )
        
        if articles['status'] != 'ok' or articles['totalResults'] == 0:
            return {
                'source': 'news',
                'articles_count': 0,
                'sentiment_score': 0,
                'sentiment_label': 'Neutral',
                'headlines': []
            }
        
        sentiments = []
        headlines = []
        
        for article in articles['articles'][:15]:
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title}. {description}"
            
            score, label = analyze_sentiment(text)
            sentiments.append(score)
            
            headlines.append({
                'title': title,
                'source': article.get('source', {}).get('name', 'Unknown'),
                'sentiment': label
            })
        
        avg_sentiment = int(np.mean(sentiments)) if sentiments else 0
        sentiment_label = "Positive" if avg_sentiment > 10 else "Negative" if avg_sentiment < -10 else "Neutral"
        
        print(f"✅ News: {sentiment_label} ({avg_sentiment}/100)")
        
        return {
            'source': 'news',
            'articles_count': len(headlines),
            'sentiment_score': avg_sentiment,
            'sentiment_label': sentiment_label,
            'headlines': headlines[:10],
            'confidence': 'high' if len(headlines) >= 5 else 'medium'
        }
        
    except Exception as e:
        print(f"❌ News error: {str(e)}")
        return {'error': str(e)}

def fetch_reddit_sentiment(stock_code):
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        return {'error': 'Reddit API not configured'}
    
    try:
        print(f"🔥 Fetching Reddit sentiment for {stock_code}...")
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        
        subreddits = ['wallstreetbets', 'stocks', 'investing']
        all_posts = []
        sentiments = []
        
        for subreddit_name in subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                for post in subreddit.search(stock_code, time_filter='week', limit=10):
                    text = f"{post.title}. {post.selftext[:200]}"
                    score, label = analyze_sentiment(text)
                    sentiments.append(score)
                    
                    all_posts.append({
                        'title': post.title,
                        'subreddit': subreddit_name,
                        'score': post.score,
                        'sentiment': label
                    })
            except:
                continue
        
        if not sentiments:
            return {
                'source': 'reddit',
                'posts_count': 0,
                'sentiment_score': 0,
                'sentiment_label': 'Neutral',
                'top_posts': []
            }
        
        avg_sentiment = int(np.mean(sentiments))
        sentiment_label = "Positive" if avg_sentiment > 15 else "Negative" if avg_sentiment < -15 else "Neutral"
        
        all_posts.sort(key=lambda x: x['score'], reverse=True)
        print(f"✅ Reddit: {sentiment_label} ({avg_sentiment}/100)")
        
        return {
            'source': 'reddit',
            'posts_count': len(all_posts),
            'sentiment_score': avg_sentiment,
            'sentiment_label': sentiment_label,
            'top_posts': all_posts[:10],
            'confidence': 'high' if len(all_posts) >= 5 else 'medium'
        }
        
    except Exception as e:
        print(f"❌ Reddit error: {str(e)}")
        return {'error': str(e)}

def combine_sentiment_sources(finnhub_sentiment, news_sentiment, reddit_sentiment):
    sources_data = []
    total_weight = 0
    weighted_score = 0
    
    if finnhub_sentiment and 'error' not in finnhub_sentiment and finnhub_sentiment.get('news_count', 0) > 0:
        weight = 0.4
        sources_data.append({
            'source': 'Finnhub',
            'sentiment': finnhub_sentiment['sentiment_label'],
            'score': finnhub_sentiment['sentiment_score'],
            'count': finnhub_sentiment['news_count'],
            'weight': '40%'
        })
        weighted_score += finnhub_sentiment['sentiment_score'] * weight
        total_weight += weight
    
    if news_sentiment and 'error' not in news_sentiment and news_sentiment.get('articles_count', 0) > 0:
        weight = 0.35
        sources_data.append({
            'source': 'News',
            'sentiment': news_sentiment['sentiment_label'],
            'score': news_sentiment['sentiment_score'],
            'count': news_sentiment['articles_count'],
            'weight': '35%'
        })
        weighted_score += news_sentiment['sentiment_score'] * weight
        total_weight += weight
    
    if reddit_sentiment and 'error' not in reddit_sentiment and reddit_sentiment.get('posts_count', 0) > 0:
        weight = 0.25
        sources_data.append({
            'source': 'Reddit',
            'sentiment': reddit_sentiment['sentiment_label'],
            'score': reddit_sentiment['sentiment_score'],
            'count': reddit_sentiment['posts_count'],
            'weight': '25%'
        })
        weighted_score += reddit_sentiment['sentiment_score'] * weight
        total_weight += weight
    
    if total_weight == 0:
        return {
            'combined_score': 0,
            'combined_label': 'Neutral',
            'confidence': 'low',
            'sources': sources_data
        }
    
    combined_score = int(weighted_score / total_weight)
    combined_label = "Positive" if combined_score > 15 else "Negative" if combined_score < -15 else "Neutral"
    
    confidence = 'very high' if len(sources_data) >= 3 else 'high' if len(sources_data) == 2 else 'medium'
    
    return {
        'combined_score': combined_score,
        'combined_label': combined_label,
        'confidence': confidence,
        'sources': sources_data,
        'finnhub_data': finnhub_sentiment if finnhub_sentiment and 'error' not in finnhub_sentiment else None,
        'news_data': news_sentiment if news_sentiment and 'error' not in news_sentiment else None,
        'reddit_data': reddit_sentiment if reddit_sentiment and 'error' not in reddit_sentiment else None
    }


# ==========================================
# AI Analysis
# ==========================================

def get_enhanced_analysis_with_predictions(stock_data, sentiment_data, prediction_data):
    if not GEMINI_API_KEY:
        return {'error': 'Gemini API key not configured'}
    
    sentiment_section = ""
    if sentiment_data and 'combined_score' in sentiment_data:
        sources_text = "\n".join([
            f"  - {s['source']}: {s['sentiment']} ({s['score']}/100)"
            for s in sentiment_data.get('sources', [])
        ])
        sentiment_section = f"\n**SENTIMENT:** {sentiment_data['combined_label']} ({sentiment_data['combined_score']}/100)\nSources:\n{sources_text}\n"
    
    prediction_section = ""
    if prediction_data and prediction_data.get('success'):
        predictions_text = "\n".join([
            f"  - {model}: {pred}"
            for model, pred in prediction_data['predictions'].items()
        ])
        prediction_section = f"\n**ML PREDICTIONS ({prediction_data['prediction_period']}):**\nEnsemble: {prediction_data['ensemble_prediction']} ({prediction_data['prediction_direction']})\n{predictions_text}\n"
    
    prompt = f"""Analyze this stock:

**Stock:** {stock_data['stock_code']} - ${stock_data['current_price']} ({stock_data['price_change_percent']:+.2f}%)
**Trend:** {stock_data['trend']}
**RSI:** {stock_data['technical_indicators']['RSI']}
**MACD:** {stock_data['technical_indicators']['MACD']}
{sentiment_section}{prediction_section}

Provide: 1) Summary 2) Technical Analysis 3) Sentiment 4) ML Predictions 5) Risk Level 6) Trading Strategy 7) Confidence"""

    try:
        url = f"{GEMINI_API_BASE}/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
        response = requests.post(url, json={
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {'temperature': 0.3, 'maxOutputTokens': 8192}
        }, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return {
                    'analysis': result['candidates'][0]['content']['parts'][0]['text'],
                    'model': 'gemini-2.0-flash-exp'
                }
        
        return {'error': 'Analysis generation failed'}
    except Exception as e:
        return {'error': str(e)}


# ==========================================
# Flask Routes
# ==========================================

@app.route('/')
def serve_index():
    """Serve main page"""
    return send_file('index.html')

@app.route('/style.css')
def serve_css():
    """Serve CSS"""
    return send_file('style.css', mimetype='text/css')

@app.route('/script.js')
def serve_js():
    """Serve JavaScript"""
    return send_file('script.js', mimetype='application/javascript')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'services': {
            'alpha_vantage': ALPHA_VANTAGE_KEY != 'demo',
            'gemini_ai': bool(GEMINI_API_KEY),
            'finnhub': bool(FINNHUB_API_KEY),
            'news_api': bool(NEWS_API_KEY),
            'reddit_api': bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)
        }
    }), 200

@app.route('/analyze_with_sentiment', methods=['POST'])
def analyze_stock_with_sentiment():
    """Main analysis endpoint"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f"Bearer {API_KEY}":
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    stock_code = data.get('stock_code')
    market_type = data.get('market_type', 'US')
    include_sentiment = data.get('include_sentiment', True)
    include_predictions = data.get('include_predictions', True)
    
    if not stock_code:
        return jsonify({'error': 'Stock code required'}), 400
    
    if ALPHA_VANTAGE_KEY == 'demo':
        return jsonify({'error': 'Alpha Vantage API key not configured'}), 400
    
    print(f"\n{'='*60}\n🔍 Analyzing {stock_code}\n{'='*60}")
    
    # Fetch stock data
    time.sleep(1)
    df, stock_data = fetch_stock_data_for_ml(stock_code, market_type)
    
    if df is None or 'error' in stock_data:
        return jsonify(stock_data), 400
    
    print(f"✅ Stock data: ${stock_data['current_price']} ({stock_data['price_change_percent']:+.2f}%)")
    
    # Sentiment
    sentiment_data = None
    if include_sentiment:
        finnhub_sentiment = fetch_finnhub_sentiment(stock_code)
        time.sleep(1)
        news_sentiment = fetch_news_sentiment(stock_code)
        time.sleep(1)
        reddit_sentiment = fetch_reddit_sentiment(stock_code)
        time.sleep(1)
        sentiment_data = combine_sentiment_sources(finnhub_sentiment, news_sentiment, reddit_sentiment)
    
    # ML Predictions
    prediction_data = None
    if include_predictions:
        print("🤖 Training ML models...")
        prediction_data = train_and_predict_stock(df, prediction_horizon=5)
        if prediction_data.get('success'):
            print(f"✅ Prediction: {prediction_data['ensemble_prediction']}")
    
    # AI Analysis
    print("🤖 Generating AI analysis...")
    ai_result = get_enhanced_analysis_with_predictions(stock_data, sentiment_data, prediction_data)
    
    if 'error' in ai_result:
        return jsonify({
            'stock_data': stock_data,
            'sentiment_data': sentiment_data,
            'prediction_data': prediction_data,
            'ai_analysis': None,
            'warning': ai_result['error']
        }), 200
    
    print("✅ Analysis complete")
    
    result = {
        'stock_data': stock_data,
        'sentiment_data': sentiment_data,
        'prediction_data': prediction_data,
        'ai_analysis': ai_result['analysis'],
        'model_info': {
            'provider': 'Google Gemini',
            'model': 'gemini-2.0-flash-exp',
            'includes_ml_predictions': include_predictions and prediction_data and prediction_data.get('success'),
            'ml_models': ['Linear Regression', 'Random Forest', 'LightGBM', 'XGBoost'] if include_predictions else []
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(result), 200

@app.route('/chart_data', methods=['POST'])
def get_chart_data():
    """Get chart data"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f"Bearer {API_KEY}":
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    stock_code = data.get('stock_code')
    market_type = data.get('market_type', 'US')
    
    if not stock_code or ALPHA_VANTAGE_KEY == 'demo':
        return jsonify({'error': 'Invalid request'}), 400
    
    try:
        symbol = get_alpha_vantage_symbol(stock_code, market_type)
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': '5min',
            'outputsize': 'full',
            'apikey': ALPHA_VANTAGE_KEY
        }
        
        response = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=10)
        chart_response = response.json()
        
        if 'Time Series (5min)' not in chart_response:
            return jsonify({'error': 'No chart data'}), 400
        
        time_series = chart_response['Time Series (5min)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = df[col].astype(float)
        df['Volume'] = df['Volume'].astype(int)
        
        df['SMA_5'] = calculate_sma(df['Close'], 5)
        df['SMA_20'] = calculate_sma(df['Close'], 20)
        
        chart_data = {
            'timestamps': df.index.strftime('%Y-%m-%d %H:%M').tolist(),
            'prices': df['Close'].round(2).tolist(),
            'volumes': df['Volume'].tolist(),
            'sma5': df['SMA_5'].fillna(0).round(2).tolist(),
            'sma20': df['SMA_20'].fillna(0).round(2).tolist()
        }
        
        return jsonify(chart_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ==========================================
# Main
# ==========================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 AI STOCK ANALYZER SERVER")
    print("="*60)
    print(f"\n📊 URL: http://localhost:8080")
    print(f"\n📡 Services:")
    print(f"   Alpha Vantage: {'✅' if ALPHA_VANTAGE_KEY != 'demo' else '❌'}")
    print(f"   Google Gemini: {'✅' if GEMINI_API_KEY else '❌'}")
    print(f"   Finnhub: {'✅' if FINNHUB_API_KEY else '❌'}")
    print(f"   NewsAPI: {'✅' if NEWS_API_KEY else '❌'}")
    print(f"   Reddit: {'✅' if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET else '❌'}")
    
    if ALPHA_VANTAGE_KEY == 'demo':
        print("\n⚠️  Alpha Vantage key required!")
        print("   Get key: https://www.alphavantage.co/support/#api-key")
    
    if not GEMINI_API_KEY:
        print("\n⚠️  Gemini API key required!")
        print("   Get key: https://ai.google.dev/")
    
    print("\n" + "="*60)
    print("Starting server...")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8080, debug=True)