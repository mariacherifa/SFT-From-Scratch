#############################################################################
#                              Tokenizer 
#############################################################################
import torch 

def load_tokenizer(checkpoint_path, device):

    checkpoint = torch.load(
           checkpoint_path, 
           map_location = device)

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    return stoi, itos

def encode(text, stoi):
    result = []

    for ch in text: 
        if ch not in stoi:
            raise ValueError(
                f"Unknown character: {repr(ch)}"
            )
        result.append(stoi[ch])

    return result 

def decode(ids, itos):
    result = ""

    for id in ids: 
        result += itos[id]

    return result 



    