import pytest
from src.preprocess import preprocess_pipeline

def test_pipeline_config_default():
    # Default: remove english=True, stop=True, stem=True
    text = "Saya feeling happy banget hari ini"
    # flow: 
    # 1. clean -> saya feeling happy banget hari ini
    # 2. norm -> saya feeling happy sangat hari ini
    # 3. eng -> removing 'feeling', 'happy'. -> saya sangat hari ini
    # 4. stop -> removing 'saya', 'sangat', 'hari', 'ini'?? 
    #    stopwords extended has 'saya', 'sangat', 'hari', 'ini'. So result might be empty?
    
    cleaned, stats = preprocess_pipeline(text)
    # Expect drastically reduced text
    assert stats['english_removed'] >= 2

def test_pipeline_no_english_removal():
    text = "Saya feeling happy"
    cleaned, stats = preprocess_pipeline(text, config={'remove_english': False, 'remove_stopwords': False, 'apply_stemming': False})
    assert "feeling" in cleaned
    assert "happy" in cleaned
    assert stats['english_removed'] == 0

def test_pipeline_slang_normalization():
    text = "aku gk mw mkan"
    # aku -> saya (norm? or whitelist?)
    # gk -> tidak
    # mw -> mau
    # mkan -> makan (stemming might affect, but normalization first)
    
    cleaned, stats = preprocess_pipeline(text, config={
        'remove_english': False,
        'remove_stopwords': False,
        'apply_stemming': False,
        'normalize_slang': True
    })
    
    assert "tidak" in cleaned
    assert "mau" in cleaned
    
def test_pipeline_stemming():
    text = "memakan dimakan"
    cleaned, stats = preprocess_pipeline(text, config={
        'remove_english': False, 
        'remove_stopwords': False, 
        'apply_stemming': True,
        'normalize_slang': False
    })
    
    assert "makan" in cleaned

def test_empty_input():
    cleaned, stats = preprocess_pipeline("")
    assert cleaned == ""
    assert stats['original_tokens'] == 0
