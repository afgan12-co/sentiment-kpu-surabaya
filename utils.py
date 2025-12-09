"""
Utility functions for text preprocessing and sentiment labeling
"""
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Slang dictionary for normalization
SLANG_DICT = {
    'jgn': 'jangan', 'lg': 'lagi', 'udah': 'sudah', 'gak': 'tidak',
    'gaada': 'tidak ada', 'org': 'orang', 'pd': 'pada', 'bgt': 'banget',
    'yg': 'yang', 'gk': 'tidak', 'skrg': 'sekarang', 'dpt': 'dapat',
    'trs': 'terus', 'utk': 'untuk', 'jd': 'jadi', 'klo': 'kalau',
    'dr': 'dari', 'dlm': 'dalam', 'bkn': 'bukan', 'sy': 'saya',
    'gue': 'saya', 'lo': 'kamu', 'aja': 'saja', 'emang': 'memang',
    'banget': 'sangat', 'gimana': 'bagaimana', 'kenapa': 'mengapa'
}

# Lexicon dictionaries for sentiment scoring
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

# Indonesian stopwords
STOPWORDS = set(stopwords.words('indonesian'))


def preprocess_text(text):
    """
    Complete text preprocessing pipeline.
    
    Steps:
    1. Case folding (lowercase)
    2. Cleaning (remove URLs, mentions, hashtags, special chars)
    3. Tokenization
    4. Normalization (slang replacement)
    5. Stopwords removal
    6. Stemming
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned and processed text
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Step 1: Case folding
    text = text.lower()
    
    # Step 2: Cleaning
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove special characters (keep only letters, numbers, and basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return ""
    
    # Step 3: Tokenization
    tokens = word_tokenize(text)
    
    # Step 4: Normalization (slang replacement)
    tokens = [SLANG_DICT.get(token, token) for token in tokens]
    
    # Step 5: Stopwords removal
    tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    
    # Step 6: Stemming
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Join back to string
    cleaned_text = ' '.join(tokens)
    
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
