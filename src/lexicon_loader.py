import json
import os
import functools
from typing import Dict, Set

# Base directory for lexicons
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICON_DIR = os.path.join(BASE_DIR, 'assets', 'lexicon')

@functools.lru_cache(maxsize=None)
def load_json_lexicon(filename: str) -> Dict[str, str]:
    """
    Load a JSON lexicon file with caching.
    
    Args:
        filename (str): Filename in assets/lexicon directory
        
    Returns:
        Dict[str, str]: Dictionary content
    """
    filepath = os.path.join(LEXICON_DIR, filename)
    try:
        if not os.path.exists(filepath):
            print(f"Warning: Lexicon file not found: {filepath}")
            return {}
            
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}

@functools.lru_cache(maxsize=None)
def load_txt_lexicon(filename: str) -> Set[str]:
    """
    Load a TXT lexicon file (one entry per line) with caching.
    
    Args:
        filename (str): Filename in assets/lexicon directory
        
    Returns:
        Set[str]: Set of entries
    """
    filepath = os.path.join(LEXICON_DIR, filename)
    try:
        if not os.path.exists(filepath):
            print(f"Warning: Lexicon file not found: {filepath}")
            return set()
            
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read lines, strip whitespace, remove empty lines
            lines = [line.strip().lower() for line in f if line.strip()]
            return set(lines)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return set()

def get_base_stops() -> Set[str]:
    """Get NLTK base Indonesian stopwords if available"""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words('indonesian'))
    except:
        return set()

def load_lexicons(force_reload: bool = False) -> Dict:
    """
    Load all required lexicons.
    
    Args:
        force_reload (bool): If True, clear cache before loading
        
    Returns:
        Dict containing all loaded lexicons
    """
    if force_reload:
        load_json_lexicon.cache_clear()
        load_txt_lexicon.cache_clear()
    
    # Load separate files
    slang = load_json_lexicon('slang_map.json')
    typo = load_json_lexicon('typo_map.json')
    
    # Load stopwords (extended + NLTK)
    stops_base = get_base_stops()
    stops_extended = load_txt_lexicon('stopwords_id_extended.txt')
    stopwords = stops_base.union(stops_extended)
    
    # Load whitelists
    indo_whitelist = load_txt_lexicon('indo_vocab_whitelist.txt')
    english_keep = load_txt_lexicon('english_keep_whitelist.txt')
    
    # Load English wordlist for detection
    english_words = load_txt_lexicon('english_common.txt')
    
    return {
        'slang': slang,
        'typo': typo,
        'stopwords': stopwords,
        'indo_whitelist': indo_whitelist,
        'english_keep': english_keep,
        'english_words': english_words
    }

def get_lexicon_stats() -> Dict[str, int]:
    """Return statistics about loaded lexicons"""
    lexicons = load_lexicons()
    return {
        'Slang Entries': len(lexicons['slang']),
        'Typo Entries': len(lexicons['typo']),
        'Stopwords': len(lexicons['stopwords']),
        'Indo Whitelist': len(lexicons['indo_whitelist']),
        'English Dictionary': len(lexicons['english_words'])
    }
