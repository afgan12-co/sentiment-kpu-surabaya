"""
Utility functions for text preprocessing and sentiment labeling
UPDATED: Now uses scr.preprocess pipeline
"""
import streamlit as st
from src.preprocess import preprocess_pipeline
from src.lexicon_loader import load_lexicons

# Lexicon dictionaries for sentiment scoring (kept here for labeling compatibility)
# These are small enough to keep defined here, or could be moved to json
LEXICON_POSITIVE = {
    'bagus': 5, 'hebat': 4, 'suka': 3, 'mantap': 4, 'senang': 3,
    'baik': 3, 'puas': 4, 'keren': 5, 'cinta': 5, 'sayang': 5,
    'ok': 3, 'oke': 3, 'positif': 4, 'sukses': 5, 'indah': 4,
    'cantik': 4, 'luar biasa': 5, 'sempurna': 5, 'menang': 4,
    'setuju': 3, 'benar': 3, 'tepat': 3, 'jujur': 4, 'adil': 4
}

LEXICON_NEGATIVE = {
    'buruk': -5, 'jelek': -4, 'benci': -5, 'sedih': -4, 'marah': -5,
    'kecewa': -5, 'parah': -5, 'gagal': -4, 'rusak': -4, 'tidak': -2,
    'salah': -3, 'korup': -5, 'curang': -5, 'bodoh': -4, 'tolol': -5,
    'gila': -3, 'hancur': -5, 'mengerikan': -5, 'jahat': -5, 'bohong': -4
}

def preprocess_text(text):
    """
    Legacy wrapper for preprocess_pipeline.
    Uses default settings (English removal ON, Stopwords ON, Stemming ON).
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned and processed text
    """
    # Use the new pipeline
    cleaned_text, _ = preprocess_pipeline(text, config={
        'remove_english': True,
        'remove_stopwords': True,
        'apply_stemming': True,
        'normalize_slang': True
    })
    return cleaned_text


def lexicon_label(text):
    """
    Label text with sentiment using lexicon-based approach.
    
    Args:
        text (str): Preprocessed text (cleaned_text)
        
    Returns:
        str: Sentiment label ('positif', 'negatif', or 'netral')
    """
    if not isinstance(text, str) or not text.strip():
        return 'netral'
    
    # Calculate sentiment score
    words = text.split()
    score = 0
    
    for word in words:
        # Add positive scores
        if word in LEXICON_POSITIVE:
            score += LEXICON_POSITIVE[word]
        # Add negative scores
        if word in LEXICON_NEGATIVE:
            score += LEXICON_NEGATIVE[word]
    
    # Determine label based on score
    if score > 0:
        return 'positif'
    elif score < 0:
        return 'negatif'
    else:
        return 'netral'


def validate_columns(df, required_cols):
    """
    Validate that dataframe has required columns.
    
    Args:
        df: pandas DataFrame
        required_cols: list of required column names
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        error_msg = f"Dataset harus memiliki kolom: {', '.join(missing)}. " \
                   f"Silakan lakukan preprocessing dan labeling terlebih dahulu."
        return False, error_msg
    
    return True, ""
