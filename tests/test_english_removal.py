import pytest
from src.preprocess import EnglishDetector, remove_english_tokens
from src.lexicon_loader import load_lexicons

@pytest.fixture
def detector():
    lexicons = load_lexicons()
    return EnglishDetector(
        lexicons['english_words'], 
        lexicons['indo_whitelist'],
        lexicons['english_keep']
    )

def test_pure_english(detector):
    tokens = ["this", "is", "a", "good", "morning"]
    cleaned, removed = remove_english_tokens(tokens, detector)
    assert removed > 0

def test_pure_indonesian(detector):
    tokens = ["saya", "makan", "nasi", "goreng", "enak"]
    cleaned, removed = remove_english_tokens(tokens, detector)
    assert removed == 0
    assert len(cleaned) == 5

def test_mixed_sentence(detector):
    # "saya feeling happy hari ini"
    tokens = ["saya", "feeling", "happy", "hari", "ini"]
    cleaned, removed = remove_english_tokens(tokens, detector)
    assert "feeling" not in cleaned
    assert "happy" not in cleaned
    assert "saya" in cleaned
    assert "hari" in cleaned

def test_user_reported_cases(detector):
    # User reported "friend", "lockdown" were not removed.
    # We expect them to be removed now.
    tokens = ["friend", "lockdown", "pandemic", "lockdowns"]
    cleaned, removed = remove_english_tokens(tokens, detector)
    assert len(cleaned) == 0
    assert removed == 4

def test_user_rant_case(detector):
    # Specific case: "milu dibawabawa fuck yall i dont even care if you are a friend or family yall who voted for him can go rot in hell"
    # Note: "milu" might be slang for "melu" (ikut), "dibawabawa" is indo.
    # We assume "milu" and "dibawabawa" are NOT in English dictionary.
    
    # Tokenizing similarly to pipeline (lower, split)
    text = "milu dibawabawa fuck yall i dont even care if you are a friend or family yall who voted for him can go rot in hell"
    tokens = text.split()
    
    cleaned, removed = remove_english_tokens(tokens, detector)
    
    # We expect "milu" and "dibawabawa" to remain
    assert "milu" in cleaned
    assert "dibawabawa" in cleaned
    
    # We expect english distinct words to be removed
    english_words = ["fuck", "yall", "dont", "care", "friend", "family", "voted", "rot", "hell", "if", "for", "him"]
    for word in english_words:
        assert word not in cleaned, f"Expected '{word}' to be removed"

def test_whitelist_preservation(detector):
    # Entities like Jokowi, KPU, Rp
    tokens = ["jokowi", "pergi", "ke", "kpu"]
    cleaned, removed = remove_english_tokens(tokens, detector)
    assert "jokowi" in cleaned
    assert "kpu" in cleaned  # whitelisted

def test_keep_technical_terms(detector):
    # "training model data" -> keep 'training', 'model', 'data' because in whitelist
    tokens = ["training", "model", "data"]
    cleaned, removed = remove_english_tokens(tokens, detector)
    assert "training" in cleaned
    assert "model" in cleaned
    assert "data" in cleaned
