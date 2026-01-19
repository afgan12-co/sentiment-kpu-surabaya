"""
Auto English Detection with Multi-Signal Approach
Detects English tokens automatically without manual input

Signals:
- Signal A: Dictionary lookup (english_common.txt)
- Signal B: Character n-gram language scoring  
- Signal C: Dataset-driven frequency analysis
"""

import re
import hashlib
from typing import List, Dict, Set, Tuple
from collections import Counter
import pandas as pd
from nltk.tokenize import word_tokenize


def normalize_apostrophe(text: str) -> str:
    """
    Normalize apostrophe variants for consistent detection
    don't → dont, can't → cant, etc.
    """
    # Remove all apostrophe variants
    text = text.replace("'", "")
    text = text.replace("'", "")
    text = text.replace("`", "")
    text = text.replace("'", "")
    return text


class MultiSignalEnglishDetector:
    """
    Detect English tokens using multiple signals for robustness
    """
    
    def __init__(self, english_words: Set[str], indo_whitelist: Set[str], 
                 english_keep: Set[str]):
        """
        Args:
            english_words: Set of English dictionary words
            indo_whitelist: Indonesian loanwords to preserve (internet, video, akun, dll)
            english_keep: Proper nouns to keep (Google, YouTube, etc.)
        """
        self.english_words = english_words
        self.indo_whitelist = indo_whitelist
        self.english_keep = english_keep
        
        # Signal B: English characteristic n-grams
        self.english_ngrams = {
            'bi': ['th', 'wh', 'sh', 'ch', 'ph'],
            'tri': ['tion', 'ght', 'ough', 'ould'],
            'end': ['ing', 'ed', 'er', 'ly', 'ty']
        }
        
        # Dataset-driven candidates (populated dynamically)
        self.dataset_candidates = set()
        
    def compute_ngram_score(self, token: str) -> float:
        """
        Signal B: Compute English likelihood based on character n-grams
        Returns score 0.0-1.0 (higher = more likely English)
        """
        token_lower = token.lower()
        score = 0.0
        indicators = 0
        
        # Check for English-specific bigrams
        for ngram in self.english_ngrams['bi']:
            if ngram in token_lower:
                score += 0.20 # Increased from 0.15
                indicators += 1
                
        # Check for English-specific trigrams
        for ngram in self.english_ngrams['tri']:
            if ngram in token_lower:
                score += 0.35 # Increased from 0.25
                indicators += 1
                
        # Check for English suffixes
        for ngram in self.english_ngrams['end']:
            if token_lower.endswith(ngram):
                score += 0.30 # Increased from 0.20
                indicators += 1
        
        # Additional English indicators
        if re.search(r'[aeiou]{3,}', token_lower): # Triple vowels are rare in ID
            score += 0.20
            indicators += 1
        
        # Normalize score
        if indicators > 0:
            score = min(score, 1.0)
            
        return score
    
    def is_english(self, token: str, use_dataset=True) -> Tuple[bool, str]:
        """
        Determine if token is English using multi-signal approach
        
        Returns:
            (is_english: bool, detection_method: str)
            detection_method: 'signal_A', 'signal_B', 'signal_C', 'whitelist', or 'none'
        """
        token_lower = token.lower()
        
        # Normalize apostrophes first
        token_normalized = normalize_apostrophe(token_lower)
        
        # Priority 1: Check Indonesian whitelist (KEEP these)
        if token_lower in self.indo_whitelist or token_normalized in self.indo_whitelist:
            return (False, 'indo_whitelist')
            
        # Priority 2: Check English keep list (brand names, proper nouns)
        if token_lower in self.english_keep or token_normalized in self.english_keep:
            return (False, 'english_keep')
        
        # Signal A: Dictionary lookup  
        if token_lower in self.english_words or token_normalized in self.english_words:
            return (True, 'signal_A')
        
        # Signal B: N-gram scoring (for longer words not in dictionary)
        if len(token_lower) >= 5:
            ngram_score = self.compute_ngram_score(token_lower)
            if ngram_score >= 0.5:  # High confidence threshold
                return (True, 'signal_B')
        
        # Signal C: Dataset-driven candidates
        if use_dataset and token_lower in self.dataset_candidates:
            return (True, 'signal_C')
            
        return (False, 'none')
    
    def detect_english_tokens(self, tokens: List[str]) -> Dict:
        """
        Detect and separate English vs Indonesian tokens
        
        Returns:
            {
                'english_tokens': List[str],
                'indonesian_tokens': List[str],
                'detection_methods': Dict[str, str],  # token -> method
                'english_count': int,
                'indonesian_count': int
            }
        """
        english_tokens = []
        indonesian_tokens = []
        detection_methods = {}
        
        for token in tokens:
            # Skip numbers
            if token.isdigit():
                indonesian_tokens.append(token)
                continue
            
            # Allow short tokens to be checked against dictionary
            if not token.strip():
                continue
                
            is_eng, method = self.is_english(token)
            
            if is_eng:
                english_tokens.append(token)
                detection_methods[token] = method
            else:
                indonesian_tokens.append(token)
        
        return {
            'english_tokens': english_tokens,
            'indonesian_tokens': indonesian_tokens,
            'detection_methods': detection_methods,
            'english_count': len(english_tokens),
            'indonesian_count': len(indonesian_tokens)
        }


def build_english_candidates_from_dataset(
    df: pd.DataFrame, 
    text_column: str = 'text',
    lexicons: Dict = None,
    top_k: int = 5000,
    min_freq: int = 3
) -> Tuple[Set[str], pd.DataFrame]:
    """
    Signal C: Auto-detect English candidates from dataset
    
    Args:
        df: DataFrame with text data
        text_column: Column name containing text
        lexicons: Dictionary of loaded lexicons
        top_k: Maximum number of candidates to return
        min_freq: Minimum frequency to consider
        
    Returns:
        (candidates_set, report_df)
    """
    if lexicons is None:
        from src.lexicon_loader import load_lexicons
        lexicons = load_lexicons()
    
    # Extract all tokens from dataset
    all_tokens = []
    for text in df[text_column]:
        if isinstance(text, str):
            # Normalize apostrophes before tokenization
            text_norm = normalize_apostrophe(text.lower())
            tokens = word_tokenize(text_norm)
            all_tokens.extend(tokens)
    
    # Count token frequencies
    token_freq = Counter(all_tokens)
    
    # Filter candidates
    candidates = []
    
    for token, freq in token_freq.most_common(top_k):
        # Skip if too short
        if len(token) < 3:
            continue
            
        # Skip if pure numbers
        if token.isdigit():
            continue
            
        # Skip if frequency too low
        if freq < min_freq:
            break
            
        # Skip if in Indonesian whitelist
        if token in lexicons.get('indo_whitelist', set()):
            continue
            
        # Skip if in Indonesian stopwords
        if token in lexicons.get('stopwords', set()):
            continue
        
        # Check if likely English
        # Simple heuristic: in English words or has English n-grams
        is_in_dict = token in lexicons.get('english_words', set())
        
        # Create temp detector for ngram scoring
        temp_detector = MultiSignalEnglishDetector(
            lexicons.get('english_words', set()),
            lexicons.get('indo_whitelist', set()),
            lexicons.get('english_keep', set())
        )
        ngram_score = temp_detector.compute_ngram_score(token)
        
        english_score = 0.0
        reasons = []
        
        if is_in_dict:
            english_score += 0.6
            reasons.append('in_dictionary')
            
        if ngram_score >= 0.3:
            english_score += ngram_score * 0.4
            reasons.append(f'ngram_score={ngram_score:.2f}')
        
        # If score high enough, add to candidates
        if english_score >= 0.5:
            candidates.append({
                'token': token,
                'frequency': freq,
                'english_score': english_score,
                'reason': ', '.join(reasons)
            })
    
    # Convert to dataframe for reporting
    report_df = pd.DataFrame(candidates)
    candidates_set = set([c['token'] for c in candidates])
    
    return candidates_set, report_df


def compute_dataset_hash(df: pd.DataFrame, text_column: str = 'text') -> str:
    """
    Compute hash of dataset for caching purposes
    """
    # Use first 100 rows and column name for hash
    sample_text = ''.join(df[text_column].head(100).astype(str).tolist())
    hash_obj = hashlib.md5(sample_text.encode())
    return hash_obj.hexdigest()


def remove_english_tokens(
    tokens: List[str],
    detector: MultiSignalEnglishDetector
) -> Dict:
    """
    Remove English tokens from list with detailed stats
    
    Returns:
        {
            'cleaned_tokens': List[str],
            'removed_tokens': List[str],
            'removed_count': int,
            'removed_ratio': float,
            'detection_breakdown': Dict[str, int]  # method -> count
        }
    """
    result = detector.detect_english_tokens(tokens)
    
    # Count detection methods
    method_counts = Counter(result['detection_methods'].values())
    
    # Calculate ratio
    total = len(tokens)
    removed = result['english_count']
    ratio = removed / total if total > 0 else 0.0
    
    return {
        'cleaned_tokens': result['indonesian_tokens'],
        'removed_tokens': result['english_tokens'],
        'removed_count': removed,
        'removed_ratio': ratio,
        'detection_breakdown': dict(method_counts)
    }
