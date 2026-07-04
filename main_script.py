# ====================================================
#                     Main Script
# ====================================================

import torch
from torch.utils.data import random_split, DataLoader

from sft_dataset import SFTDataset
from model import MiniGPT
from sft_train import train
from prompt_formatting import format_example
from tokenizer import encode, decode 


# ====================================================
# Hyperparameters
# ====================================================

file_location = "/Users/mariacherifa/Desktop/SFT-From-Scratch/instruction_response.json"

block_size = 128
batch_size = 32

n_embd = 64
num_heads = 4
n_layers = 5
dropout = 0.2

n_epochs = 100
learning_rate = 1e-3
eval_interval = 10
eval_iters = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================================================
# Dataset
# ====================================================

sft_dataset = SFTDataset(
    path=file_location,
    block_size=block_size
)

vocab_size = sft_dataset.size_vocabulary
itos = sft_dataset.itos
stoi = sft_dataset.stoi

n = len(sft_dataset)
train_size = int(0.8 * n)
val_size = n - train_size

train_dataset, val_dataset = random_split(
    sft_dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    sft_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    sft_dataset,
    batch_size=batch_size,
    shuffle=False
)

# ====================================================
# Model
# ====================================================

model = MiniGPT(
    n_embd=n_embd,
    num_heads=num_heads,
    vocab_size=vocab_size,
    block_size=block_size,
    n_layers=n_layers,
    dropout=dropout
).to(device)


# ====================================================
# Training
# ====================================================

train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    eval_interval=eval_interval,
    eval_iters=eval_iters,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    device=device
)

# ====================================================
# Generation 
# ====================================================

instruction = "Translate hello to French."
prompt = format_example(instruction, None)
context =  torch.tensor(
    [encode(prompt, stoi)],
    dtype=torch.long,
    device=device)


with torch.no_grad():
    generated_ids = model.generate(context, max_new_tokens=10)

generated_text = decode(generated_ids[0].tolist(), itos)

print(generated_text)
