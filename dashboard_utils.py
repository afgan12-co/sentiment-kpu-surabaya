"""
Dashboard utility functions untuk KPU Admin Dashboard
"""

import json
import os
from datetime import datetime
from typing import List, Dict
from collections import Counter
from config import CATEGORY_KEYWORDS


# ========== CATEGORY DETECTION ==========

def detect_category(text: str) -> str:
    """
    Deteksi kategori berdasarkan keyword matching
    
    Args:
        text: Cleaned text (sudah dipreprocess)
        
    Returns:
        Category: 'kinerja', 'netralitas', atau 'kebijakan'
    """
    if not text or not isinstance(text, str):
        return 'kinerja'  # default
    
    text_lower = text.lower()
    words = text_lower.split()
    
    # Count matches untuk setiap kategori
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for word in words if word in keywords)
        scores[category] = score
    
    # Return kategori dengan skor tertinggi
    if max(scores.values()) == 0:
        return 'kinerja'  # default jika tidak ada match
    
    return max(scores, key=scores.get)


# ========== KEYWORD EXTRACTION ==========

def extract_keywords(texts: List[str], top_n: int = 10) -> List[Dict]:
    """
    Extract top N keywords dari list of texts
    
    Args:
        texts: List of cleaned texts
        top_n: Jumlah top keywords yang diinginkan
        
    Returns:
        List of {word, count} sorted by count
    """
    if not texts:
        return []
    
    # Gabungkan semua text dan split menjadi words
    all_words = []
    for text in texts:
        if isinstance(text, str):
            all_words.extend(text.split())
    
    # Count frequency
    word_counts = Counter(all_words)
    
    # Get top N
    top_keywords = word_counts.most_common(top_n)
    
    return [
        {"word": word, "count": count}
        for word, count in top_keywords
    ]


# ========== DATA PERSISTENCE ==========

def ensure_dashboard_dir():
    """Create dashboard data directory if not exists"""
    os.makedirs('data/dashboard', exist_ok=True)


def save_analysis_result(data: Dict) -> None:
    """
    Save single analysis result to JSON storage
    
    Args:
        data: dict containing analysis result
    """
    ensure_dashboard_dir()
    filepath = 'data/dashboard/results.json'
    
    # Load existing data
    results = load_analysis_results()
    
    # Add new result with auto-increment ID
    data['id'] = len(results) + 1 if results else 1
    data['timestamp'] = datetime.now().isoformat() + 'Z'
    
    results.append(data)
    
    # Save back
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_analysis_results() -> List[Dict]:
    """
    Load all analysis results from JSON storage
    
    Returns:
        List of analysis results
    """
    ensure_dashboard_dir()
    filepath = 'data/dashboard/results.json'
    
    if not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return []


# ========== STATISTICS AGGREGATION ==========

def aggregate_statistics(results: List[Dict]) -> Dict:
    """
    Generate statistics untuk dashboard dari analysis results
    
    Args:
        results: List of analysis results
        
    Returns:
        Formatted statistics dict matching spec
    """
    if not results:
        return {
            "total_data": 0,
            "last_updated": datetime.now().isoformat() + 'Z',
            "sentiment_distribution": {
                "positive": 0,
                "negative": 0,
                "neutral": 0
            },
            "categories": {}
        }
    
    # Overall statistics
    total_data = len(results)
    last_updated = max([r.get('timestamp', '') for r in results]) if results else datetime.now().isoformat() + 'Z'
    
    # Sentiment distribution (overall)
    sentiment_counts = Counter([r.get('sentiment', 'neutral') for r in results])
    sentiment_distribution = {
        "positive": sentiment_counts.get('positif', 0) + sentiment_counts.get('positive', 0),
        "negative": sentiment_counts.get('negatif', 0) + sentiment_counts.get('negative', 0),
        "neutral": sentiment_counts.get('netral', 0) + sentiment_counts.get('neutral', 0)
    }
    
    # Group by category
    categories_data = {}
    for category in ['kinerja', 'netralitas', 'kebijakan']:
        category_results = [r for r in results if r.get('category') == category]
        
        if category_results:
            # Sentiment distribution for this category
            cat_sentiment_counts = Counter([r.get('sentiment', 'neutral') for r in category_results])
            
            # Trend data (group by date)
            trend_data = calculate_trend(category_results)
            
            # Top keywords
            texts = [r.get('cleaned_text', '') for r in category_results if r.get('cleaned_text')]
            top_keywords = extract_keywords(texts, top_n=10)
            
            # Recent data (latest 5)
            recent = sorted(category_results, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]
            recent_data = [
                {
                    "id": r.get('id', 0),
                    "text": r.get('text', ''),
                    "sentiment": normalize_sentiment(r.get('sentiment', 'neutral')),
                    "confidence": r.get('confidence', 0.0),
                    "timestamp": r.get('timestamp', '')
                }
                for r in recent
            ]
            
            categories_data[category] = {
                "total": len(category_results),
                "positive": cat_sentiment_counts.get('positif', 0) + cat_sentiment_counts.get('positive', 0),
                "negative": cat_sentiment_counts.get('negatif', 0) + cat_sentiment_counts.get('negative', 0),
                "neutral": cat_sentiment_counts.get('netral', 0) + cat_sentiment_counts.get('neutral', 0),
                "sentiment_trend": trend_data,
                "top_keywords": top_keywords,
                "recent_data": recent_data
            }
        else:
            categories_data[category] = {
                "total": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "sentiment_trend": [],
                "top_keywords": [],
                "recent_data": []
            }
    
    return {
        "total_data": total_data,
        "last_updated": last_updated,
        "sentiment_distribution": sentiment_distribution,
        "categories": categories_data
    }


def calculate_trend(results: List[Dict]) -> List[Dict]:
    """
    Calculate sentiment trend by date
    
    Args:
        results: List of analysis results
        
    Returns:
        List of trend data by date
    """
    # Group by date
    date_groups = {}
    for r in results:
        timestamp = r.get('timestamp', '')
        if timestamp:
            # Extract date (YYYY-MM-DD)
            date = timestamp.split('T')[0] if 'T' in timestamp else timestamp[:10]
            if date not in date_groups:
                date_groups[date] = []
            date_groups[date].append(r)
    
    # Calculate sentiment counts per date
    trend_data = []
    for date in sorted(date_groups.keys())[-30:]:  # Last 30 days
        day_results = date_groups[date]
        sentiment_counts = Counter([r.get('sentiment', 'neutral') for r in day_results])
        
        trend_data.append({
            "date": date,
            "positive": sentiment_counts.get('positif', 0) + sentiment_counts.get('positive', 0),
            "negative": sentiment_counts.get('negatif', 0) + sentiment_counts.get('negative', 0),
            "neutral": sentiment_counts.get('netral', 0) + sentiment_counts.get('neutral', 0)
        })
    
    return trend_data


def normalize_sentiment(sentiment: str) -> str:
    """Normalize sentiment labels to English"""
    mapping = {
        'positif': 'positive',
        'negatif': 'negative',
        'netral': 'neutral'
    }
    return mapping.get(sentiment.lower(), sentiment)
