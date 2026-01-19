import os

# Path to lexicons
LEXICON_DIR = r"assets/lexicon"

def update_lexicon(filename, words):
    path = os.path.join(LEXICON_DIR, filename)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        existing_words = set(line.strip().lower() for line in f if line.strip())
        
    new_words = [w.lower() for w in words if w.lower() not in existing_words]
    
    if new_words:
        with open(path, 'a', encoding='utf-8') as f:
            for w in new_words:
                f.write(f"\n{w}")
        print(f"Added {len(new_words)} words to {filename}")
    else:
        print(f"No new words to add to {filename}")

eng_words = [
    'vote', 'voted', 'voter', 'care', 'fuck', 'fucking', 'fuckin', 'yall', "y'all", 
    'lockdown', 'tanzania', 'dont', 'doesnt', 'didnt', 'isnt', 'arent', 'wasnt', 
    'werent', 'couldnt', 'shouldnt', 'wouldnt', 'wont', 'cant', 'even', 'friend', 
    'family', 'rot', 'hell', 'i', 'the', 'is', 'are', 'was', 'were', 'who', 'for', 
    'him', 'can', 'go', 'or', 'at', 'with', 'if', 'this', 'that', 'from', 'by'
]

indo_words = [
    'milu', 'kpps', 'konoha', 'bilang', 'orang', 'tinggal', 'nyoblos', 'jg', 
    'emang', 'tuh', 'maut', 'nyawa', 'renggut', 'biasa2', 'potret', 'pilih', 
    'kasih', 'nyoblos', 'dibawabawa', 'pun', 'kok', 'sih', 'lah', 'deh'
]

if __name__ == "__main__":
    update_lexicon('english_common.txt', eng_words)
    update_lexicon('indo_vocab_whitelist.txt', indo_words)
    print("Lexicons updated successfully.")
