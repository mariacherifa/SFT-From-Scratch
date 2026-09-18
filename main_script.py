###########################################################################
#                                  Main Script
###########################################################################

import torch
from torch.utils.data import random_split, DataLoader

from sftdataset import SFTDataset
from SFT_train import train
from prepare_data import format_example
from tokenizer import encode
from model import MiniGPT


# ====================================================
# Helper: generate only the answer
# ====================================================

def generate_answer(
    model,
    prompt,
    stoi,
    itos,
    device,
    block_size,
    max_new_tokens=20
):
    model.eval()

    idx = torch.tensor(
        [encode(prompt, stoi)],
        dtype=torch.long,
        device=device
    )

    generated_chars = []

    with torch.no_grad():

        for _ in range(max_new_tokens):

            idx_context = idx[:, -block_size:]

            logits, _ = model(idx_context)

            logits = logits[:, -1, :]

            next_id = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

            next_char = itos[int(next_id.item())]

            generated_chars.append(next_char)

            idx = torch.cat(
                (idx, next_id),
                dim=1
            )

            # Stop AFTER generating the period
            if next_char == ".":
                break

    return "".join(generated_chars)

# ====================================================
# Paths
# ====================================================

dataset_path = (
    "/Users/mariacherifa/Desktop/"
    "SFT_from_GPT/dataset_SFT.json"
)

checkpoint_path = (
    "/Users/mariacherifa/Desktop/"
    "SFT_from_GPT/minigpt_pretrained_params.pt"
)

best_sft_path = (
    "/Users/mariacherifa/Desktop/"
    "SFT_from_GPT/best_sft_model.pt"
)

final_sft_path = (
    "/Users/mariacherifa/Desktop/"
    "SFT_from_GPT/minigpt_sft.pt"
)


# ====================================================
# Device
# ====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ====================================================
# Load pretrained checkpoint
# ====================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

vocab_size = checkpoint["vocab_size"]
block_size = checkpoint["block_size"]
n_embd = checkpoint["n_embd"]
num_heads = checkpoint["num_heads"]
n_layers = checkpoint["num_layers"]
dropout = checkpoint["dropout"]

print("Pretrained block size:", block_size)
print("Vocabulary size:", vocab_size)


# ====================================================
# SFT hyperparameters
# ====================================================

batch_size = 4

# Maximum number of epochs.
# Early stopping can stop before this.
n_epochs = 20

learning_rate = 1e-4

eval_interval = 20
eval_iters = 5

patience = 8
min_delta = 0.0


# ====================================================
# Build SFT Dataset
# ====================================================

sft_dataset = SFTDataset(
    path=dataset_path,
    block_size=block_size,
    stoi=stoi,
    itos=itos
)

print("\nFinal SFT dataset size:", len(sft_dataset))


# ====================================================
# Train / validation split
# ====================================================

n = len(sft_dataset)

train_size = int(0.8 * n)
val_size = n - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    sft_dataset,
    [train_size, val_size],
    generator=generator
)

print("Training examples:", len(train_dataset))
print("Validation examples:", len(val_dataset))


# ====================================================
# DataLoaders
# ====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ====================================================
# Initialize MiniGPT from pretrained checkpoint
# ====================================================

model = MiniGPT(
    n_embd=n_embd,
    num_heads=num_heads,
    vocab_size=vocab_size,
    block_size=block_size,
    n_layers=n_layers,
    dropout=dropout
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("\nPretrained MiniGPT weights loaded successfully.")


# ====================================================
# Supervised Fine-Tuning
# ====================================================

model = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    eval_interval=eval_interval,
    eval_iters=eval_iters,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    device=device,
    patience=patience,
    min_delta=min_delta,
    best_model_path=best_sft_path
)


# ====================================================
# Save final SFT checkpoint
#
# Important:
# train() should already have restored the best
# validation weights before returning the model.
# ====================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
        "block_size": block_size,
        "n_embd": n_embd,
        "num_heads": num_heads,
        "num_layers": n_layers,
        "dropout": dropout
    },
    final_sft_path
)

print("\nSFT model saved.")


# ====================================================
# Test generation on TRAINING example
# ====================================================

original_idx = train_dataset.indices[0]

test_example = sft_dataset.load_examples[
    original_idx
]

prompt, true_response = format_example(
    test_example
)

generated_answer = generate_answer(
    model=model,
    prompt=prompt,
    stoi=stoi,
    itos=itos,
    device=device,
    block_size=block_size,
    max_new_tokens=20
)

print("\n=================================")
print("TRAINING EXAMPLE")
print("=================================")

print("\nPROMPT:")
print(prompt)

print("\nTRUE RESPONSE:")
print(true_response.strip())

print("\nGENERATED RESPONSE:")
print(generated_answer)


# ====================================================
# Test generation on VALIDATION example
# ====================================================

original_idx = val_dataset.indices[0]

test_example = sft_dataset.load_examples[
    original_idx
]

prompt, true_response = format_example(
    test_example
)

generated_answer = generate_answer(
    model=model,
    prompt=prompt,
    stoi=stoi,
    itos=itos,
    device=device,
    block_size=block_size,
    max_new_tokens=20
)

print("\n=================================")
print("VALIDATION EXAMPLE")
print("=================================")

print("\nPROMPT:")
print(prompt)

print("\nTRUE RESPONSE:")
print(true_response.strip())

print("\nGENERATED RESPONSE:")
print(generated_answer)


# ====================================================
# Test on NEW unseen instruction
# ====================================================

new_example = {
    "instruction": "What is 3 + 5?",
    "input": "",
    "output": ""
}

prompt, _ = format_example(
    new_example
)

generated_answer = generate_answer(
    model=model,
    prompt=prompt,
    stoi=stoi,
    itos=itos,
    device=device,
    block_size=block_size,
    max_new_tokens=20
)

print("\n=================================")
print("NEW EXAMPLE")
print("=================================")

print("\nPROMPT:")
print(prompt)

print("\nGENERATED RESPONSE:")
print(generated_answer)