###################################################################
#                            SFT Dataset 
###################################################################

import torch 
from tokenizer import load_tokenizer, encode, decode 
from prepare_data import load_sft_data, format_example, build_sft_text

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self,path,block_size,itos,stoi):
        super().__init__()

        self.load_examples = load_sft_data(path)
        self.block_size = block_size

        self.itos = itos 
        self.stoi = stoi

        self.vocabulary_size = len(stoi)


    def __len__(self):
        return len(self.load_examples)
    
    def __getitem__(self, idx):

        example = self.load_examples[idx]

        prompt, response = format_example(example)

        full_text = prompt + response

        full_ids = encode(full_text, self.stoi)
        prompt_ids = encode(prompt, self.stoi)

        x = full_ids[:-1]
        y = full_ids[1:]

        loss_mask = [0]*(len(prompt_ids)-1)
        loss_mask += [1]*(len(y)-len(loss_mask))

        if len(x) > self.block_size:
            x = x[:self.block_size]
            y = y[:self.block_size]

            loss_mask = loss_mask[:self.block_size]

        else: 
            pad = self.block_size - len(x)
            x = x + [0]*pad
            y = y + [0]*pad
            loss_mask = loss_mask + [0]*pad


        return (torch.tensor(x, dtype= torch.long),
                torch.tensor(y, dtype = torch.long),
                torch.tensor(loss_mask, dtype = torch.float)
                )