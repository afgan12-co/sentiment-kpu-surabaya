from src.pipeline import run_pipeline, PipelineConfig

# Test cases
test_texts = [
    "saya vote untuk kandidat bagus",
    "dont care about this issue",
    "dia bilang don worry be happy",
    "pilpres vote care dont untuk rakyat",
    "internet dan video aplikasi bagus",
    "fuck aplikasi ini jelek banget yall", # Test profanity and slang
    "ini adalah a test untuk sistem kami", # Test single char English 'a'
    "lockdown di tanzania renggut nyawa" # Test new words
]

print("="*60)
print("TESTING UNIFIED PIPELINE (ENGLISH DETECTION)")
print("="*60)

config = PipelineConfig(
    remove_english=True,
    remove_stopwords=False,
    apply_stemming=False,
    normalize_slang=True
)

for i, text in enumerate(test_texts, 1):
    print(f"\n[Test {i}] Input: {text}")
    
    result = run_pipeline(text, config)
    
    print(f"Output: {result.text_final}")
    print(f"English removed ({result.english_removed_count}): {result.english_removed_tokens}")
    print(f"Detection methods: {result.english_detection_methods}")
    print("-" * 60)

print("\nTest completed!")
