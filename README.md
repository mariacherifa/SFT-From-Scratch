# SFT From Scratch 

This project builds directly on my previous repository MiniGPT, where I implemented a decoder-only Transformer language model from scratch and trained it for next-token prediction on the Tiny Shakespeare dataset.

The objective of this repository is not to build a new architecture, but to understand how a pretrained language model is transformed into an instruction-following assistant.

Instead of training on raw text, the model is fine-tuned on instruction-response pairs using the Supervised Fine-Tuning (SFT) objective.

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

# Code Walkthrough

This repository implements the complete **Supervised Fine-Tuning (SFT)** pipeline from scratch. The Transformer architecture is exactly the same as in the previous **MiniGPT** project. The main difference lies in how the training data is prepared and how the training loss is computed.

The implementation is organized into the following steps.

---

## 1. Instruction-Response Dataset

Unlike MiniGPT, which was trained on continuous text (Tiny Shakespeare), Supervised Fine-Tuning requires a dataset composed of **instruction-response pairs**.

Each example is stored as a JSON object:

```json
{
    "instruction": "Translate hello to French.",
    "response": "Bonjour."
}
```

This dataset represents the desired behavior that we want the language model to learn.

---

## 2. Prompt Formatting

Each instruction-response pair is converted into a single training sequence.

For example,

```text
Instruction:
Translate hello to French.

Response:
Bonjour.
```

During inference, only the instruction is provided:

```text
Instruction:
Translate hello to French.

Response:
```

and the model generates the response autoregressively.

---

## 3. Vocabulary Construction and Tokenization

As in the MiniGPT project, we first build a character-level vocabulary from the complete SFT corpus.

The tokenizer provides three utilities:

- vocabulary construction
- text encoding
- text decoding

allowing us to convert raw text into integer token IDs and reconstruct the original text after generation.

---

## 4. Building the SFT Dataset

The dataset is implemented as a custom `torch.utils.data.Dataset`. For every instruction-response pair, we construct:

- the input sequence (`x`)
- the shifted target sequence (`y`)
- a **response-only loss mask**

This dataset is then wrapped inside PyTorch `DataLoader`s for mini-batch training.

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

# Repository Structure

```text
SFT-From-Scratch/
├── instruction_response.json   # Instruction-response dataset used for supervised fine-tuning
├── tokenizer.py                # Vocabulary construction, encoding and decoding utilities
├── prompt_formatting.py        # Converts instruction-response pairs into training prompts
├── sft_dataset.py              # Builds the SFT dataset
├── model.py                    # MiniGPT architecture with masked cross-entropy for supervised fine-tuning
├── sft_train.py                # Training loop, validation loop and loss estimation
└── main_script.py              # Main script for training and text generation
```

Before running the project, simply update the path to the instruction dataset in `main_script.py`:

```python
file_location = "path/to/instruction_response.json"
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


