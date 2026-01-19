import os

# Path to lexicons
LEXICON_DIR = r"assets/lexicon"

def clean_and_update(filename, to_add, to_remove):
    path = os.path.join(LEXICON_DIR, filename)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        words = [line.strip().lower() for line in f if line.strip()]
        
    original_count = len(words)
    
    # Remove unwanted
    words = [w for w in words if w not in to_remove]
    
    # Add new
    for w in to_add:
        if w.lower() not in words:
            words.append(w.lower())
            
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words))
        
    print(f"File {filename}: {original_count} -> {len(words)} entries.")

# 1. Cleanup English Dict
eng_add = ['vote', 'voted', 'voter', 'care', 'fuck', 'fucking', 'fuckin', 'yall', "y'all", 'lockdown', 'tanzania', 'dont', 'even', 'friend', 'family', 'rot', 'hell', 'i', 'the', 'is', 'are', 'was', 'were', 'who', 'for', 'him', 'can', 'go', 'or', 'at', 'with', 'if', 'this', 'that', 'from', 'by', 'about', 'issue', 'friend', 'family', 'are', 'can', 'go', 'rot', 'hell']
eng_remove = ['milu']

# 2. Cleanup Indo Whitelist
indo_add = ['milu', 'kpps', 'konoha', 'bilang', 'orang', 'tinggal', 'nyoblos', 'jg', 'emang', 'tuh', 'maut', 'nyawa', 'renggut', 'biasa2', 'potret', 'pilih', 'kasih', 'dibawabawa', 'pun', 'kok', 'sih', 'lah', 'deh']
indo_remove = ['vote', 'care']

if __name__ == "__main__":
    clean_and_update('english_common.txt', eng_add, eng_remove)
    clean_and_update('indo_vocab_whitelist.txt', indo_add, indo_remove)
    print("Lexicon cleanup complete.")
