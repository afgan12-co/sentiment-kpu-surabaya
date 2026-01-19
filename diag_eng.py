from src.lexicon_loader import load_lexicons
from src.auto_english_detector import MultiSignalEnglishDetector

lexicons = load_lexicons()
detector = MultiSignalEnglishDetector(
    lexicons['english_words'],
    lexicons['indo_whitelist'],
    lexicons['english_keep']
)

tokens = ['milu', 'vote', 'fuck', 'yall', 'dont', 'i', 'the', 'kpps', 'konoha']

print(f"Indo Whitelist size: {len(lexicons['indo_whitelist'])}")
print(f"English Dict size: {len(lexicons['english_words'])}")
print(f"Is 'milu' in whitelist? {'milu' in lexicons['indo_whitelist']}")
print(f"Is 'milu' in english dict? {'milu' in lexicons['english_words']}")

for t in tokens:
    is_eng, method = detector.is_english(t)
    print(f"Token: '{t}' -> is_english: {is_eng} (Method: {method})")
