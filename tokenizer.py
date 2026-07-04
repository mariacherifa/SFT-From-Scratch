# ====================================================
#                  Tokenization 
# ====================================================

# Vocabulary building 

def build_vocab(text): 
    vocabulary = sorted(set(text))
    vocabulary_size = len(vocabulary)
    stoi = {}
    itos = {}

    for i, ch in enumerate(vocabulary):
        stoi[ch] = i 
        itos[i] = ch 

    return vocabulary, vocabulary_size, itos,stoi 

# Encoding function 

def encode(text, stoi):
    result = []

    for ch in text: 
        result.append(stoi[ch])
    
    return result

# Decoding function

def decode(ids, itos): 
    result =""

    for i in ids: 
        result += itos[int(i)]
    
    return result 
    

 