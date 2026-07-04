# =================================================
#                        SFT dataset
# =================================================
import torch
from prompt_formatting import load_json, format_example, build_sft_text
from tokenizer import build_vocab, encode, decode

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, path, block_size):
        super().__init__()

        self.load_examples= load_json(path)
        self.all_text = build_sft_text(self.load_examples)

        self.vocabulary, self.size_vocabulary, self.itos, self.stoi  = build_vocab(self.all_text)
        
        self.block_size = block_size 

    def __len__(self):
        return len(self.load_examples)
    
    def __getitem__(self,idx):
        example = self.load_examples[idx]
        prompt = format_example(example["instruction"],None)

        full_text = format_example(example["instruction"],example["response"])

        full_ids = encode(full_text, self.stoi)
        prompt_ids = encode(prompt, self.stoi)

        x = full_ids[:-1]
        y = full_ids[1:]
        loss_mask = [0]* (len(prompt_ids)-1)  
        loss_mask += [1]* (len(y)-len(loss_mask))

        if len(x) > self.block_size :
            x = x[: self.block_size]
            y = y[: self.block_size]
            loss_mask = loss_mask[: self.block_size]
        
        else : 
            pad = self.block_size - len(x)
            x = x + [0]*pad 
            y = y + [0]*pad
            loss_mask = loss_mask + [0]*pad
        
        return (
        torch.tensor(x, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(loss_mask, dtype=torch.float),
    )








