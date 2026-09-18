###################################################################
#                            Transformer Model 
###################################################################

import torch 
import torch.nn as nn 
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self,n_embd,head_size,block_size,dropout):
        super().__init__()

        self.query = nn.Linear(n_embd,head_size)
        self.key = nn.Linear(n_embd,head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
    "mask",
    torch.tril(torch.ones(block_size, block_size)),
    persistent=False
)

    def forward(self,x): 
        B,T,C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1)
        wei = wei *(k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.mask[:T, :T] == 0,float("-inf")
)
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)

        out = wei @ v

        return out 

# Multi head attention 
class MultiHeadAttention(nn.Module):
    def __init__(self,n_embd,num_heads, block_size,dropout):
        super().__init__()

        head_size = n_embd // num_heads

        self.heads = nn.ModuleList(Head(n_embd,head_size,block_size,dropout) for _ in range(num_heads))
        self.dropout = nn.Dropout(dropout) 
        self.proj = nn.Linear(head_size*num_heads , n_embd)

    def forward(self, x): 
        out = torch.cat([head(x) for head in self.heads], dim = -1)
        out = self.proj(out)
        out = self.dropout(out)

        return out 

# MLP layer 

class MLP(nn.Module): 
    def __init__(self,n_embd, dropout):
        super().__init__()

        self.nets = nn.Sequential(nn.Linear(n_embd, 4*n_embd), 
                                  nn.ReLU(), 
                                  nn.Linear(4*n_embd, n_embd))

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.dropout(self.nets(x))

# Block Transformer 

class BlockTransformer(nn.Module):
    def __init__(self,n_embd,num_heads, block_size,dropout):
        super().__init__()

        self.mha = MultiHeadAttention(n_embd,num_heads, block_size,dropout)
        self.mlp = MLP(n_embd, dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x): 
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x 

# MiniGPT model 
class MiniGPT(nn.Module):
    def __init__(self,n_embd, num_heads, vocab_size, block_size, n_layers, dropout):
        super().__init__()

        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = nn.Embedding(block_size, n_embd)

        self.blocks= nn.Sequential(*[BlockTransformer(n_embd,num_heads, block_size,dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.lnf = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self,idx, target = None, loss_mask = None): 

        B,T = idx.shape 

        tok_embd = self.token_embedding(idx)
        pos = torch.arange(T, device = idx.device)
        pos_embd = self.positional_embedding(pos)

        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.lnf(x)

        logits = self.lm_head(x)

        if loss_mask is None: 
            if target is None: 
                loss = None 

            else: 
                B,T,C = logits.shape
                logits_flat = logits.reshape(B*T,C)
                target_flat = target.reshape(B*T)
                loss = F.cross_entropy(logits_flat, target_flat)

        else: 
            if target is None: 
                loss = None 

            else: 
                B,T,C = logits.shape 
                logits_flat = logits.reshape(B*T,C)
                target_flat = target.reshape(B*T)
                loss_mask_flat = loss_mask.reshape(B*T).float()
                loss_per_token = F.cross_entropy(logits_flat,target_flat,reduction="none")
                denominator = loss_mask_flat.sum().clamp_min(1.0)
                loss = (loss_per_token * loss_mask_flat).sum()/ denominator 

        return logits, loss 

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            idx_context = idx[:, -self.block_size:]

            logits, loss = self(idx_context)
            temperature = 0.7
            logits = logits[:,-1,:]
            logits = logits/temperature

            probs = F.softmax(logits, dim = -1)
            idx_new = torch.argmax(logits, dim=-1, keepdim=True)
            #torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx,idx_new), dim =1)

        return idx

    





    
    


    