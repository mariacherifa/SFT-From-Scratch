# SFT From Scratch 

This repository implements **Supervised Fine-Tuning (SFT) from scratch** on top of a pretrained decoder-only Transformer language model.

It builds directly on my previous **MiniGPT** project, where I implemented a small GPT-style Transformer from scratch in PyTorch and pretrained it with causal next-token prediction on the TinyStories dataset.

The goal of this repository is not to build a larger language model or introduce a new Transformer architecture.

Instead, the objective is to understand, at implementation level, how a pretrained language model is transformed into an **instruction-following model**.

# From MiniGPT to Supervised Fine-Tuning

In **MiniGPT**, the model was trained on continuous text, for example:

```text
The sky is blue.
```

The objective was simply to predict the next token:

```text
The  → sky
sky  → is
is   → blue
...
```

Every token in the sequence contributed equally to the training loss. The model learned the statistical structure of language by performing **next-token prediction** over raw text.

Supervised Fine-Tuning (SFT) introduces a different training paradigm. Instead of learning from continuous text, the model is trained on **instruction-response pairs**:

```text
Instruction:
Translate "hello" into French.

Response:
Bonjour.
```

The model still performs autoregressive next-token prediction, but now the objective is different:

- the **instruction** provides the context,
- the **response** is the desired output,
- and the loss is computed **only on the response tokens**.

The instruction is therefore used to condition the model's predictions, but it does not contribute to the optimization objective.

This response-only masked loss is the key difference between standard language-model pretraining and supervised fine-tuning. It allows a pretrained language model to transition from predicting arbitrary text to learning how to follow user instructions.

# Pretrained Model Initialization
SFT does not train a new Transformer from random initialization.

The MiniGPT architecture is instantiated exactly as during pretraining, and all pretrained parameters are restored from the saved checkpoint. SFT then continues optimizing these pretrained parameters on instruction-response data.


# Code Walkthrough

This repository implements the complete **Supervised Fine-Tuning (SFT)** pipeline from scratch. The Transformer architecture is exactly the same as in the previous **MiniGPT** project. The main difference lies in how the training data is prepared and how the training loss is computed.

The implementation is organized into the following steps.

---

## 1. Instruction-Response Dataset

Unlike MiniGPT, which was trained on continuous text (Tiny Shakespeare), Supervised Fine-Tuning requires a dataset composed of **instruction-response pairs**.

Each example is stored as a JSON object:

```json
{
    "instruction": "What is 1 + 2?",
    "input": "",
    "output": "3."
  }
```

This dataset represents the desired behavior that we want the language model to learn.

---

## 2. Prompt Formatting

Each instruction-response pair is converted into a single training sequence.

For example,

```text
Instruction:
"What is 1 + 2 ?"

Response:
"3".
```

During inference, only the instruction is provided:

```text
Instruction:
"What is 1 + 2 ?".

Response:
```

and the model generates the response autoregressively.

---

## 3. Tokenizer reuse

An important detail is that SFT must use the same vocabulary and token IDs as pretraining.

The tokenizer mappings are therefore loaded directly from the pretrained checkpoint. 

A new vocabulary is not constructed from the SFT dataset.

This is necessary because each row of the pretrained embedding matrix already corresponds to a specific token ID.

Changing the token-to-ID mapping would destroy the semantic alignment between the tokenizer and the pretrained embedding weights.

The project currently uses a character-level tokenizer, inherited from MiniGPT.

---

## 5. Response-Only Loss Mask

Unlike standard language modeling, we do **not** compute the loss on the entire sequence.

For the example

```text
Instruction:
Translate hello to French.

Response:
Bonjour.
```
the instruction is used only as context, while the optimization is performed exclusively on the response tokens. Conceptually,

```text
Instruction:
○ ○ ○ ○ ○

Response:
● ● ● ●
```

where

- ○ = ignored during optimization
- ● = contributes to the training loss

This masking strategy is the defining characteristic of Supervised Fine-Tuning.

---

## 6. MiniGPT Architecture

The language model itself is unchanged from the previous repository. The architecture contains:

- token embeddings
- positional embeddings
- multi-head causal self-attention
- feed-forward networks
- residual connections
- layer normalization
- language modeling head

Since the objective of this repository is to understand SFT, the focus is placed on the training pipeline rather than on modifying the Transformer architecture.

---

## 7. Supervised Fine-Tuning

The forward pass computes the logits exactly as in MiniGPT.

However, instead of averaging the cross-entropy over every token, we compute the loss independently for each token and multiply it by the response mask before averaging.

This trains the model to generate the response while conditioning on the instruction.

---

## 8. Training and Evaluation

The model is optimized using **AdamW**.

During training, we periodically evaluate both the training and validation losses to monitor convergence and detect overfitting.

---

## 9. Autoregressive Generation

After fine-tuning, the model can answer instructions by generating one token at a time.

Given

```text
Instruction:
What is the capital of France?

Response:
```

the model predicts the response autoregressively until the answer is complete.

# Why Use a Small Controlled Dataset?

A major lesson from this project is that SFT quality depends strongly on the capability of the pretrained model.

MiniGPT is intentionally tiny:

- character-level tokenization
- small embedding dimension
- few Transformer layers
- short context window
- lightweight pretraining

It should therefore not be expected to behave like a modern instruction-tuned LLM.

Using large instruction datasets such as Alpaca introduces tasks that are far beyond the capacity and context length of this model.

For this reason, the final SFT experiment uses a smaller and more structured dataset.

The goal is to demonstrate:

pretrained language model
        ↓
instruction-conditioned training
        ↓
response masking
        ↓
supervised fine-tuning
        ↓
instruction-conditioned generation

rather than maximize benchmark performance.

# Repository Structure

```text
SFT_from_GPT/
│
├── dataset_SFT.json
│   └── Short instruction-response dataset
│
├── tokenizer.py
│   └── Loads pretrained vocabulary and provides encode/decode utilities
│
├── prepare_data.py
│   └── Loads and formats instruction-response examples
│
├── sftdataset.py
│   └── Builds x, y and response-only loss masks
│
├── model.py
│   └── MiniGPT Transformer + masked SFT loss
│
├── SFT_train.py
│   └── Training, validation and early stopping
│
├── main_script.py
│   └── Loads pretrained checkpoint, fine-tunes and evaluates MiniGPT
│
└── minigpt_pretrained_params.pt
    └── Pretrained MiniGPT checkpoint
```

Before running the project, simply update the path to the instruction dataset in `main_script.py`:

```python
file_location = "path/to/dataset_SFT.json.json"
```

Then run the entire training pipeline with:

```bash
python main_script.py
```

# Main difference from MiniGPT 

| MiniGPT                             | SFT-From-Scratch                            |
| ----------------------------------- | ------------------------------------------- |
| Trained on raw text                 | Trained on instruction-response pairs       |
| Every token contributes to the loss | Only response tokens contribute to the loss |
| Learns general language modeling    | Learns instruction following                |
| Tiny Shakespeare dataset            | JSON instruction dataset                    |
| Standard cross-entropy              | Masked cross-entropy                        |


