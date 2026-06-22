# Character-Level Transformer Language Model (Java)

A GPT-style decoder-only Transformer trained on German Wikipedia, implemented **from scratch in pure Java** — no ML frameworks, no autodiff library.

The goal is to understand and implement every component by hand: attention, backpropagation, Adam, Layer Norm.
I'm aware that Java is probably the worst language performance-wise to build anything ML-related, but for me it was the perfect choice for learning the fundamentals from scratch. Java gives you all the mathematical support you need while forcing you to think explicitly about how things work and fit together. Building this project was essential for developing a deeper understanding of the GPT-style model architecture, and it laid a solid foundation to continue in other languages like Python using high-level libraries like PyTorch or NumPy.

---

## Architecture

```
Input characters
       │
EmbeddingLayer  (vocabSize × embDim)
       +
Positional Embeddings  (seqLen × embDim, learned)
       │
  ┌────┴────┐
  │ TransformerBlock × N │   (Pre-Norm, causal)
  │                      │
  │  LayerNorm            │
  │  MultiHeadAttention   │   scaled dot-product, causal mask
  │  residual +           │
  │  LayerNorm            │
  │  FFN (Linear-ReLU-Linear, 4× hidden) │
  │  residual +           │
  └──────────────────────┘
       │
 Linear Layer  (embDim → vocabSize)
       │
 SoftmaxCCE loss / Temperature Sampling
```

### Model Sizes

| Config | Embedding | Heads | Blocks | Parameters |
|--------|-----------|-------|--------|------------|
| 1M     | 64        | 4     | 2      | ~1M        |
| 5M     | 192       | 8     | 6      | ~5M        |
| 25M    | 384       | 8     | 12     | ~25M       |

---

## Implementation Details

### What's implemented by hand

| Component | File | Notes |
|-----------|------|-------|
| Matrix ops + parallel multiply | `math/` | Zero-allocation hot paths, parallel dispatch for ops > 500K flops |
| Multi-head self-attention | `network/MultiHeadAttention.java` | Causal masking, scaled dot-product |
| Backprop through attention | `network/SelfAttentionLayer.java` | Full softmax Jacobian, not diagonal approx |
| Layer Normalization | `network/LayerNorm.java` | Forward + backward with mean/variance gradients |
| Adam optimizer | `optimizer/Adam.java` | Bias-corrected m̂/v̂ |
| Softmax + CCE (fused) | `loss/SoftmaxCCE.java` | Avoids gradient suppression from naive separate passes |
| Xavier & He initialization | `initialization/` | Normal and uniform variants |
| Activation functions | `activation/` | ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, Linear |
| Additional losses | `loss/` | MSE, BinaryCrossEntropy, FocalLoss, Huber |

### Key design decisions

- **Pre-Norm** (LayerNorm before attention/FFN) for training stability
- **Fused SoftmaxCCE backward**: combined gradient `p - y` avoids the vanishing gradient problem of computing Softmax and CCE separately
- **Learned positional embeddings** instead of sinusoidal
- **Gradient accumulation** per batch via `+=` (not overwrite), correct for mini-batch SGD
- **Cached weight transpose** in dense layers to avoid repeated allocation in the backward pass

---

## Project Structure

```
src/
├── Main.java                     # Training loop & text generation
├── math/
│   ├── Matrix.java               # Core matrix operations
│   └── MatrixMultiplication.java # Optimized, parallelized multiply
├── network/
│   ├── TransformerBlock.java     # Pre-Norm block: MHA + FFN + residuals
│   ├── MultiHeadAttention.java   # Head splitting, concat, output projection
│   ├── SelfAttentionLayer.java   # Single head: QKV, causal mask, softmax
│   ├── LayerNorm.java            # γ/β learned, full backward pass
│   ├── EmbeddingLayer.java       # Token → embedding lookup
│   └── Layer.java                # Dense layer (Linear)
├── activation/                   # ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, Linear
├── loss/                         # MSE, BCE, CCE, Huber, FocalLoss, SoftmaxCCE
├── optimizer/                    # Adam (SGD stub)
├── initialization/               # Xavier (uniform/normal), He (uniform/normal)
├── tokenizer/                    # Character-level tokenizer
├── data/                         # Text loading & sequence windowing
└── utils/                        # Memory profiling
```

---

## Training

### Prerequisites

- Java 17+
- Training corpus as `train.txt` in the project root (UTF-8 text)

For German Wikipedia, download a dump and use the included `parse_wiki.py` to extract clean text:

```bash
python parse_wiki.py dewiki-*.xml > train.txt
```

### Run

Open `src/Main.java` and set:

```java
static final String CONFIG = "1M";   // "1M", "5M" or "25M"
static final int    TRAIN_STEPS = 50_000;
static final double LR          = 3e-4;
```

Compile and run from the project root:

```bash
javac -d out src/**/*.java src/Main.java
java -cp out Main
```

The program will ask whether to **train** or **generate** text.

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `SEQ_LEN` | 64 | Context window (characters) |
| `BATCH_SIZE` | 32 | Samples per gradient step |
| `LR` | 3e-4 | Adam learning rate |
| `TEMPERATURE` | 0.8 | Sampling temperature (0 = greedy, >1 = more random) |
| `LOG_EVERY` | 200 | Steps between loss printouts |

---

## Motivation

This project was built to deeply understand how Transformers work at the level of individual matrix operations and gradient flows — before relying on frameworks like PyTorch or JAX. Implementing backpropagation through attention, layer norm, and the fused softmax-CCE loss by hand forces a level of precision that framework usage alone does not require.
