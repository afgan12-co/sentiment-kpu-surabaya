"""
Sentiment Lexicon System
Loads sentiment dictionaries and computes sentiment scores with negation/intensifier handling
"""
import os
from typing import Dict, List, Set, Tuple, NamedTuple
from functools import lru_cache
import pandas as pd

class SentimentLexicons(NamedTuple):
    """Container for all sentiment lexicons"""
    positive: Set[str]
    negative: Set[str]
    neutral: Set[str]
    negation: Set[str]
    intensifier: Set[str]

class ValidationResult(NamedTuple):
    """Result of lexicon validation"""
    is_valid: bool
    conflicts: List[Tuple[str, List[str]]]  # (word, [lexicons it appears in])
    warnings: List[str]

class SentimentResult(NamedTuple):
    """Result of sentiment computation"""
    pos_count: int
    neg_count: int
    neu_count: int
    negation_count: int
    intensifier_count: int
    sentiment_score: float
    sentiment_label: str
    details: str  # Human-readable explanation

@lru_cache(maxsize=1)
def load_sentiment_lexicons() -> SentimentLexicons:
    """
    Load all sentiment lexicon files with caching
    
    Returns:
        SentimentLexicons object with all loaded word sets
    """
    base_path = os.path.join("assets", "sentiment_lexicon")
    
    def load_txt_set(filename: str) -> Set[str]:
        """Load a text file into a set of words"""
        filepath = os.path.join(base_path, filename)
        if not os.path.exists(filepath):
            return set()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            words = set()
            for line in f:
                word = line.strip().lower()
                if word and not word.startswith('#'):  # Skip comments
                    words.add(word)
            return words
    
    return SentimentLexicons(
        positive=load_txt_set("positive.txt"),
        negative=load_txt_set("negative.txt"),
        neutral=load_txt_set("neutral.txt"),
        negation=load_txt_set("negation.txt"),
        intensifier=load_txt_set("intensifier.txt")
    )

def validate_lexicons(lex: SentimentLexicons) -> ValidationResult:
    """
    Validate lexicons for conflicts and issues
    
    Args:
        lex: SentimentLexicons object
        
    Returns:
        ValidationResult with conflicts and warnings
    """
    conflicts = []
    warnings = []
    
    # Check overlaps between positive, negative, neutral
    lexicon_map = {
        'positive': lex.positive,
        'negative': lex.negative,
        'neutral': lex.neutral
    }
    
    # Find words appearing in multiple sentiment categories
    all_words = set()
    word_sources = {}
    
    for lex_name, lex_words in lexicon_map.items():
        for word in lex_words:
            if word not in word_sources:
                word_sources[word] = []
            word_sources[word].append(lex_name)
    
    # Identify conflicts (words in 2+ categories)
    for word, sources in word_sources.items():
        if len(sources) > 1:
            conflicts.append((word, sources))
    
    # Warnings
    if len(lex.positive) < 100:
        warnings.append(f"Positive lexicon kecil: hanya {len(lex.positive)} kata")
    if len(lex.negative) < 100:
        warnings.append(f"Negative lexicon kecil: hanya {len(lex.negative)} kata")
    
    is_valid = len(conflicts) == 0
    
    return ValidationResult(
        is_valid=is_valid,
        conflicts=conflicts,
        warnings=warnings
    )

def compute_sentiment(
    tokens: List[str],
    lexicons: SentimentLexicons = None,
    handle_negation: bool = True,
    handle_intensifier: bool = True,
    negation_window: int = 3
) -> SentimentResult:
    """
    Compute sentiment score for a list of tokens
    
    Args:
        tokens: List of preprocessed tokens  
        lexicons: SentimentLexicons (if None, will load)
        handle_negation: Apply negation flipping
        handle_intensifier: Apply intensifier boosting
        negation_window: Look-back window for negation (default 3 tokens)
        
    Returns:
        SentimentResult with counts and label
    """
    if lexicons is None:
        lexicons = load_sentiment_lexicons()
    
    pos_count = 0
    neg_count = 0
    neu_count = 0
    negation_count = 0
    intensifier_count = 0
    
    score = 0.0
    details_parts = []
    
    # Track recent context for negation detection
    recent_negation = False
    recent_intensifier = False
    negation_tracker = []  # Track positions
    
    for i, token in enumerate(tokens):
        token_lower = token.lower()
        
        # Check if this is a negation word
        if token_lower in lexicons.negation:
            negation_count += 1
            negation_tracker.append(i)
            recent_negation = True
            continue
        
        # Check if this is an intensifier
        if token_lower in lexicons.intensifier:
            intensifier_count += 1
            recent_intensifier = True
            continue
        
        # Check sentiment
        is_positive = token_lower in lexicons.positive
        is_negative = token_lower in lexicons.negative
        is_neutral = token_lower in lexicons.neutral
        
        # Determine if within negation window
        in_negation_context = False
        if handle_negation and negation_tracker:
            # Check if any recent negation is within window
            for neg_pos in negation_tracker:
                if i - neg_pos <= negation_window:
                    in_negation_context = True
                    break
        
        # Apply sentiment scoring
        if is_positive:
            if in_negation_context:
                # Flip: positive becomes negative
                neg_count += 1
                base_score = -1
                details_parts.append(f"'{token}' (pos→neg by negation)")
            else:
                pos_count += 1
                base_score = 1
                details_parts.append(f"'{token}' (pos)")
        elif is_negative:
            if in_negation_context:
                # Flip: negative becomes positive
                pos_count += 1
                base_score = 1
                details_parts.append(f"'{token}' (neg→pos by negation)")
            else:
                neg_count += 1
                base_score = -1
                details_parts.append(f"'{token}' (neg)")
        elif is_neutral:
            neu_count += 1
            base_score = 0
            # Don't add to details for neutral (too verbose)
        else:
            # Unknown word
            base_score = 0
        
        # Apply intensifier boost
        if handle_intensifier and recent_intensifier and base_score != 0:
            base_score *= 1.5
            recent_intensifier = False  # Reset after use
        
        score += base_score
        
        # Reset negation if outside window
        if negation_tracker:
            negation_tracker = [pos for pos in negation_tracker if i - pos <= negation_window]
    
    # Determine label
    if score > 0:
        sentiment_label = 'positif'
    elif score < 0:
        sentiment_label = 'negatif'
    else:
        sentiment_label = 'netral'
    
    # Create human-readable details
    details = f"Score={score:.1f} (pos:{pos_count}, neg:{neg_count}, neu:{neu_count})"
    if negation_count > 0:
        details += f", negasi:{negation_count}"
    if intensifier_count > 0:
        details += f", intensifier:{intensifier_count}"
    
    return SentimentResult(
        pos_count=pos_count,
        neg_count=neg_count,
        neu_count=neu_count,
        negation_count=negation_count,
        intensifier_count=intensifier_count,
        sentiment_score=score,
        sentiment_label=sentiment_label,
        details=details
    )

def audit_misclassified_tokens(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    lexicons: SentimentLexicons = None
) -> Dict:
    """
    Audit tokens that frequently appear in wrong sentiment classes
    
    Args:
        df: DataFrame with text and labels
        text_col: Name of column with tokenized text (space-separated)
        label_col: Name of column with sentiment labels
        lexicons: SentimentLexicons (if None, will load)
        
    Returns:
        Dict with audit results
    """
    if lexicons is None:
        lexicons = load_sentiment_lexicons()
    
    # Track token appearances by label
    token_label_counts = {}
    
    for _, row in df.iterrows():
        text = str(row[text_col])
        label = str(row[label_col])
        
        tokens = text.split()
        for token in tokens:
            token_lower = token.lower()
            if token_lower not in token_label_counts:
                token_label_counts[token_lower] = {'positif': 0, 'negatif': 0, 'netral': 0}
            
            if label in token_label_counts[token_lower]:
                token_label_counts[token_lower][label] += 1
    
    # Find mismatches
    neg_in_netral = []
    pos_in_negatif = []
    neg_in_positif = []
    
    for token, counts in token_label_counts.items():
        total = sum(counts.values())
        if total < 3:  # Skip rare tokens
            continue
        
        # Check if negative word appears mostly in netral docs
        if token in lexicons.negative:
            netral_ratio = counts['netral'] / total if total > 0 else 0
            if netral_ratio > 0.5:
                neg_in_netral.append((token, counts['netral'], netral_ratio))
        
        # Check if positive word appears in negatif docs
        if token in lexicons.positive:
            negatif_ratio = counts['negatif'] / total if total > 0 else 0
            if negatif_ratio > 0.3:
                pos_in_negatif.append((token, counts['negatif'], negatif_ratio))
        
        # Check if negative word appears in positif docs
        if token in lexicons.negative:
            positif_ratio = counts['positif'] / total if total > 0 else 0
            if positif_ratio > 0.3:
                neg_in_positif.append((token, counts['positif'], positif_ratio))
    
    # Sort by count
    neg_in_netral.sort(key=lambda x: x[1], reverse=True)
    pos_in_negatif.sort(key=lambda x: x[1], reverse=True)
    neg_in_positif.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'neg_words_in_netral_docs': neg_in_netral[:30],
        'pos_words_in_negatif_docs': pos_in_negatif[:30],
        'neg_words_in_positif_docs': neg_in_positif[:30]
    }
