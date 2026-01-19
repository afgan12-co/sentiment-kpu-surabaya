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
            
    # Sort for cleanliness
    words.sort()
            
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words))
        
    print(f"File {filename}: {original_count} -> {len(words)} entries.")

# 1. Comprehensive English Word List (Signal A)
# Adding all words from user's leakage report + common English function words
eng_add = [
    'vote', 'voted', 'voter', 'voting', 'care', 'cared', 'fuck', 'fucking', 'fuckin', 
    'shit', 'damn', 'yall', "y'all", 'lockdown', 'tanzania', 'dont', 'doesnt', 'didnt', 
    'isnt', 'arent', 'wasnt', 'werent', 'couldnt', 'shouldnt', 'wouldnt', 'wont', 'cant', 
    'even', 'friend', 'family', 'rot', 'hell', 'i', 'the', 'is', 'are', 'was', 'were', 
    'who', 'for', 'him', 'can', 'go', 'or', 'at', 'with', 'if', 'this', 'that', 'from', 
    'by', 'about', 'issue', 'you', 'your', 'my', 'me', 'we', 'our', 'they', 'them', 
    'his', 'her', 'it', 'its', 'an', 'to', 'in', 'on', 'of', 'and', 'but', 'as', 
    'be', 'do', 'will', 'have', 'has', 'had', 'been', 'which', 'where', 'when', 
    'why', 'how', 'there', 'here', 'all', 'any', 'some', 'no', 'not', 'up', 'down', 
    'out', 'so', 'then', 'just', 'more', 'most', 'very', 'still', 'now', 'only'
]

# 2. Indonesian Whitelist (Ensure these are NOT removed)
indo_add = [
    'milu', 'kpps', 'konoha', 'bilang', 'orang', 'tinggal', 'nyoblos', 'jg', 
    'emang', 'tuh', 'maut', 'nyawa', 'renggut', 'biasa2', 'potret', 'pilih', 
    'kasih', 'dibawabawa', 'pun', 'kok', 'sih', 'lah', 'deh', 'dong', 'kan'
]

if __name__ == "__main__":
    clean_and_update('english_common.txt', eng_add, ['milu'])
    clean_and_update('indo_vocab_whitelist.txt', indo_add, ['vote', 'care'])
    print("Lexicon synchronization complete.")
