import re
import string
from collections import Counter
from typing import List, Dict, Tuple, Set

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize

from src.lexicon_loader import load_lexicons

# Initialize stemmer (cached module-level)
_stemmer = None

def get_stemmer():
    global _stemmer
    if _stemmer is None:
        factory = StemmerFactory()
        _stemmer = factory.create_stemmer()
    return _stemmer

class EnglishDetector:
    """
    Helper class to detect and filter English tokens.
    Uses a combination of wordlist lookup and character n-gram heuristics.
    """
    def __init__(self, english_words: Set[str], indo_whitelist: Set[str], keep_list: Set[str]):
        # Keep ALL words in the english list, even short ones (e.g. 'i', 'if', 'or')
        self.english_words = english_words 
        self.indo_whitelist = indo_whitelist
        self.keep_list = keep_list
        
        # English-specific char sequences (simplified)
        self.eng_ngrams = ['th', 'sh', 'ght', 'tion', 'sion', 'ough', 'ould', 'wh', 'ea', 'ee', 'ou']
        
    def is_english(self, token: str) -> bool:
        """
        Determine if a token is likely English.
        """
        token_lower = token.lower()
        
        # 1. Check whitelists (Always keep these)
        if token_lower in self.indo_whitelist:
            return False
        if token_lower in self.keep_list:
            return False
            
        # 2. Check strict English dictionary
        if token_lower in self.english_words:
            return True
            
        # 3. Heuristic: Check for characteristic English n-grams
        # Only for longer words that aren't in dictionary
        if len(token_lower) > 4:
            ngram_score = 0
            for ngram in self.eng_ngrams:
                if ngram in token_lower:
                    ngram_score += 1
            
            # High threshold for heuristic to avoid false positives
            if ngram_score >= 2: 
                return True
                
        return False

def clean_text_basic(text: str) -> str:
    """Basic cleaning: URL, mentions, specific chars"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special chars but keep punctuation for tokenization context if needed
    # For this pipeline, we remove most special chars but keep letters/numbers
    # We allow hyphens for compound words
    text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def remove_english_tokens(tokens: List[str], detector: EnglishDetector) -> Tuple[List[str], int]:
    """
    Remove English tokens from list.
    Returns (cleaned_tokens, count_removed)
    """
    cleaned = []
    removed_count = 0
    
    for token in tokens:
        # Don't check numbers 
        if token.isdigit():
            cleaned.append(token)
            continue
            
        # Previously we skipped len < 2, but 'i' and 'a' need checking if strict
        # The detector.is_english checks whitelists first, so 'a' might be kept if not in english list (wait 'a' IS in english list)
        # But 'a' is not a word in Indonesian (except slang?). 'ke' is. 
        # If 'a' is in English list, it will be removed. 
        # Indonesian single letters usually don't mean much unless context.
        
        if detector.is_english(token):
            removed_count += 1
        else:
            cleaned.append(token)
            
    return cleaned, removed_count

def preprocess_pipeline(text: str, config: Dict = None) -> Tuple[str, Dict]:
    """
    Main preprocessing pipeline.
    
    Args:
        text: Input string
        config: Configuration dictionary with keys:
            - remove_english (bool): Default True
            - remove_stopwords (bool): Default True
            - apply_stemming (bool): Default True
            - normalize_slang (bool): Default True
            
    Returns:
        (cleaned_text, stats_dict)
    """
    if config is None:
        config = {}
        
    cfg_english = config.get('remove_english', True)
    cfg_stop = config.get('remove_stopwords', True)
    cfg_stem = config.get('apply_stemming', True)
    cfg_slang = config.get('normalize_slang', True)
    
    stats = {
        'original_tokens': 0,
        'final_tokens': 0,
        'english_removed': 0,
        'stopwords_removed': 0
    }
    
    if not isinstance(text, str) or not text.strip():
        return "", stats
        
    # Load resources
    lexicons = load_lexicons()
    english_detector = EnglishDetector(
        lexicons['english_words'], 
        lexicons['indo_whitelist'],
        lexicons['english_keep']
    )
    
    # 1. Cleaning & Case Folding
    text_clean = clean_text_basic(text)
    
    # 2. Tokenization
    tokens = word_tokenize(text_clean)
    stats['original_tokens'] = len(tokens)
    
    # 3. Normalization (Slang & Typo)
    if cfg_slang:
        normalized_tokens = []
        for token in tokens:
            # Check typo map first
            token = lexicons['typo'].get(token, token)
            # Check slang map
            token = lexicons['slang'].get(token, token)
            normalized_tokens.append(token)
        tokens = normalized_tokens
    
    # 4. English Removal (Before stopwords to ensure English stopwords are caught if any)
    # Actually best to remove English content words. English stopwords might be same as Indo? 
    # Usually 'in', 'on', 'at' are English stopwords. 'di', 'ke' are Indo.
    # We remove English first.
    if cfg_english:
        tokens, removed_count = remove_english_tokens(tokens, english_detector)
        stats['english_removed'] = removed_count
    
    # 5. Stopwords Removal
    if cfg_stop:
        initial_count = len(tokens)
        tokens = [t for t in tokens if t not in lexicons['stopwords']]
        stats['stopwords_removed'] = initial_count - len(tokens)
        
    # 6. Stemming
    if cfg_stem:
        stemmer = get_stemmer()
        tokens = [stemmer.stem(t) for t in tokens]
        
    # Final assembly
    final_text = ' '.join(tokens)
    stats['final_tokens'] = len(tokens)
    
    return final_text, stats
