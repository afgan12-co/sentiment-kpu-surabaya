"""
Unified Preprocessing Pipeline - Single Source of Truth (SSOT)
All pages must use this module for preprocessing
"""
import re
from typing import List, Dict, NamedTuple
from dataclasses import dataclass, field

from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from src.lexicon_loader import load_lexicons
from src.preprocess import clean_text_basic, get_stemmer
from src.auto_english_detector import (
    MultiSignalEnglishDetector, 
    normalize_apostrophe,
    remove_english_tokens as remove_eng_fn
)
from src.sentiment_lexicon import (
    load_sentiment_lexicons,
    compute_sentiment,
    SentimentResult
)

@dataclass
class PipelineConfig:
    """Configuration for pipeline execution"""
    remove_english: bool = True
    remove_stopwords: bool = True
    apply_stemming: bool = True
    normalize_slang: bool = True
    compute_sentiment: bool = False  # Whether to compute sentiment labels
    handle_negation: bool = True  # For sentiment computation
    handle_intensifier: bool = True  # For sentiment computation

@dataclass
class PipelineResult:
    """
    Complete result of pipeline with full lineage
    All intermediate states are preserved for transparency
    """
    # Original
    text_raw: str
    
    # Stage 1: Basic cleaning
    text_clean: str
    
    # Stage 2: English removal
    text_id_only: str
    
    # Stage 3: Normalization (slang/typo)
    text_normalized: str
    
    # Stage 4: Tokenization & stemming
    text_final: str = ""  # Final joined text
    
    english_removed_count: int = 0
    english_removed_ratio: float = 0.0
    english_removed_tokens: List[str] = field(default_factory=list)
    english_detection_methods: Dict[str, str] = field(default_factory=dict)
    
    tokens_raw: List[str] = field(default_factory=list)  # After tokenization, before stopword/stem
    tokens_final: List[str] = field(default_factory=list)  # After all processing
    
    stopwords_removed_count: int = 0
    
    # Sentiment (optional)
    sentiment: SentimentResult = None
    
    # Metadata
    config: PipelineConfig = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame storage"""
        d = {
            'text_raw': self.text_raw,
            'text_clean': self.text_clean,
            'text_id_only': self.text_id_only,
            'text_normalized': self.text_normalized,
            'text_final': self.text_final,
            'tokens_final': ' '.join(self.tokens_final),  # Store as string
            'english_removed_count': self.english_removed_count,
            'english_removed_ratio': self.english_removed_ratio,
            'stopwords_removed_count': self.stopwords_removed_count
        }
        
        if self.sentiment:
            d.update({
                'pos_count': self.sentiment.pos_count,
                'neg_count': self.sentiment.neg_count,
                'neu_count': self.sentiment.neu_count,
                'sentiment_score': self.sentiment.sentiment_score,
                'sentiment_label': self.sentiment.sentiment_label
            })
        
        return d

def run_pipeline(text: str, config: PipelineConfig = None) -> PipelineResult:
    """
    Execute the complete preprocessing pipeline
    
    Pipeline Stages:
    1. Basic Cleaning (URL, email, special chars)
    2. English Removal (Multi-Signal)
    3. Normalization (slang, typo)
    4. Tokenization
    5. Stopword Removal
    6. Stemming
    7. Sentiment Labeling (optional)
    
    Args:
        text: Raw input text
        config: PipelineConfig (uses defaults if None)
        
    Returns:
        PipelineResult with complete lineage
    """
    if config is None:
        config = PipelineConfig()
    
    # Initialize result
    result = PipelineResult(
        text_raw=text,
        text_clean="",
        text_normalized="",
        text_id_only="",
        config=config
    )
    
    # Handle empty input
    if not isinstance(text, str) or not text.strip():
        return result
    
    # Load resources
    lexicons = load_lexicons()
    detector = MultiSignalEnglishDetector(
        lexicons['english_words'],
        lexicons['indo_whitelist'],
        lexicons['english_keep']
    )
    
    # === STAGE 1: Basic Cleaning ===
    result.text_clean = clean_text_basic(text)
    
    # === STAGE 2: English Removal ===
    # CRITICAL: We do this before slang normalization to prevent English slang
    # (like "dont") from being normalized into Indonesian.
    if config.remove_english:
        # We use normalize_apostrophe here to catch variants like "don't" -> "dont"
        text_for_eng = normalize_apostrophe(result.text_clean)
        tokens_for_eng = word_tokenize(text_for_eng)
        
        eng_result = remove_eng_fn(tokens_for_eng, detector)
        
        result.text_id_only = ' '.join(eng_result['cleaned_tokens'])
        result.english_removed_count = eng_result['removed_count']
        result.english_removed_ratio = eng_result['removed_ratio']
        result.english_removed_tokens = eng_result['removed_tokens']
        result.english_detection_methods = eng_result['detection_breakdown']
    else:
        result.text_id_only = result.text_clean
    
    # === STAGE 3: Normalization (Slang & Typo) ===
    if config.normalize_slang:
        # Tokenize id_only text for normalization
        temp_tokens = word_tokenize(result.text_id_only)
        normalized_tokens = []
        
        for token in temp_tokens:
            # Check typo first, then slang
            token = lexicons['typo'].get(token, token)
            token = lexicons['slang'].get(token, token)
            normalized_tokens.append(token)
        
        result.text_normalized = ' '.join(normalized_tokens)
    else:
        result.text_normalized = result.text_id_only
    
    # Save raw tokens (after cleaning/eng-removal/normalization, but before stopword/stem)
    result.tokens_raw = word_tokenize(result.text_normalized)
    
    # === STAGE 4: Tokenization & Stopword Removal ===
    tokens = result.tokens_raw.copy()
    
    if config.remove_stopwords:
        initial_count = len(tokens)
        tokens = [t for t in tokens if t not in lexicons['stopwords']]
        result.stopwords_removed_count = initial_count - len(tokens)
    
    # === STAGE 5: Stemming ===
    if config.apply_stemming:
        stemmer = get_stemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    
    # === STAGE 6: Finalize ===
    result.tokens_final = tokens
    result.text_final = ' '.join(tokens)
    
    # === STAGE 7: Sentiment Labeling (Optional) ===
    if config.compute_sentiment and len(result.tokens_final) > 0:
        sentiment_lex = load_sentiment_lexicons()
        result.sentiment = compute_sentiment(
            result.tokens_final,
            sentiment_lex,
            handle_negation=config.handle_negation,
            handle_intensifier=config.handle_intensifier
        )
    
    return result

def validate_no_english_leakage(
    tokens: List[str],
    english_detector: MultiSignalEnglishDetector,
    sample_size: int = 50
) -> Dict:
    """
    Validate that no English words leaked through
    
    Args:
        tokens: List of final tokens
        english_detector: EnglishDetector instance
        sample_size: Number of tokens to check
        
    Returns:
        Dict with validation results
    """
    leaked = []
    
    # Sample tokens to check (don't check all for performance)
    check_tokens = tokens[:sample_size] if len(tokens) > sample_size else tokens
    
    for token in check_tokens:
        if token.isdigit():
            continue
        is_eng, _ = english_detector.is_english(token)
        if is_eng:
            leaked.append(token)
    
    return {
        'has_leakage': len(leaked) > 0,
        'leaked_tokens': leaked,
        'leaked_count': len(leaked),
        'tokens_checked': len(check_tokens)
    }
