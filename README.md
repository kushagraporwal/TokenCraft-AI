# LLM Model Implementation - Documentation

This project implements a GPT-style Large Language Model (LLM) from scratch, including tokenization, architecture, pretraining, and fine-tuning capabilities.

## Project Structure

The project is organized into modular Python files, each with comprehensive documentation:

### Core Modules

1. **`tokenizer.py`** - Text tokenization utilities
   - `SimpleTokenizerV1`: Basic tokenizer with vocabulary mapping
   - `SimpleTokenizerV2`: Enhanced tokenizer with unknown token handling
   - `BPETokenizer`: Byte Pair Encoding tokenizer (GPT-2 style)
   - `create_vocab_from_text()`: Vocabulary creation utility

2. **`attention.py`** - Attention mechanisms
   - `SelfAttention_v1`: Basic self-attention with learnable parameters
   - `SelfAttention_v2`: Improved self-attention with nn.Linear
   - `CausalAttention`: Causal (masked) attention for autoregressive models
   - `MultiHeadAttentionWrapper`: Multi-head attention (sequential)
   - `MultiHeadAttention`: Efficient multi-head attention (parallel)

3. **`transformer.py`** - Transformer architecture components
   - `GELU`: Gaussian Error Linear Unit activation
   - `LayerNorm`: Layer normalization
   - `FeedForward`: Feedforward neural network
   - `TransformerBlock`: Core transformer block
   - `GPTModel`: Complete GPT model architecture

4. **`data_loader.py`** - Data loading utilities
   - `GPTDatasetV1`: Dataset for pretraining (next-token prediction)
   - `SpamDataset`: Dataset for classification fine-tuning
   - `InstructionDataset`: Dataset for instruction fine-tuning
   - `create_dataloader_v1()`: DataLoader creation utility

5. **`training.py`** - Training utilities
   - `calculate_loss()`: Cross-entropy loss computation
   - `generate_text()`: Text generation with sampling
   - `train_epoch()`: Single epoch training
   - `evaluate()`: Model evaluation

6. **`classification.py`** - Classification fine-tuning
   - `ClassificationHead`: Classification layer
   - `GPTForClassification`: GPT model with classification head
   - `calculate_classification_loss()`: Classification loss
   - `calculate_accuracy()`: Accuracy computation
   - `train_classification_epoch()`: Classification training
   - `evaluate_classification()`: Classification evaluation

7. **`config.py`** - Model configurations
   - `GPT_CONFIG_124M`: Configuration for 124M parameter model
   - `get_model_size()`: Parameter count estimation

## Quick Start

### 1. Tokenization

```python
from tokenizer import BPETokenizer

tokenizer = BPETokenizer()
text = "Hello, world!"
token_ids = tokenizer.encode(text)
decoded = tokenizer.decode(token_ids)
```

### 2. Model Creation

```python
from transformer import GPTModel
from config import GPT_CONFIG_124M

model = GPTModel(GPT_CONFIG_124M)
```

### 3. Training

```python
from data_loader import create_dataloader_v1
from training import train_epoch, calculate_loss
import torch.optim as optim

# Create dataloader
dataloader = create_dataloader_v1(
    text, batch_size=8, max_length=256, stride=128
)

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train
loss = train_epoch(model, dataloader, optimizer)
```

### 4. Text Generation

```python
from training import generate_text
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
generated = generate_text(
    model, tokenizer, "The future of AI is",
    max_new_tokens=50, temperature=0.8
)
```

### 5. Classification Fine-tuning

```python
from classification import GPTForClassification, train_classification_epoch
from data_loader import SpamDataset
from torch.utils.data import DataLoader
import tiktoken

# Load pretrained model
gpt_model = GPTModel(GPT_CONFIG_124M)
# Load pretrained weights here...

# Add classification head
model = GPTForClassification(gpt_model, num_classes=2)
model.freeze_gpt()  # Freeze pretrained weights

# Create dataset
tokenizer = tiktoken.get_encoding("gpt2")
dataset = SpamDataset("spam_data.csv", tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Train
optimizer = optim.Adam(model.classification_head.parameters(), lr=1e-3)
loss, accuracy = train_classification_epoch(model, dataloader, optimizer)
```

## Key Concepts Explained

### Tokenization
- **Simple Tokenizers**: Split text on whitespace/punctuation, map to vocabulary
- **BPE Tokenizer**: Subword tokenization that handles out-of-vocabulary words
- **Vocabulary**: Mapping between tokens and integer IDs

### Attention Mechanisms
- **Self-Attention**: Each position attends to all positions in the sequence
- **Causal Attention**: Prevents attending to future positions (for autoregressive models)
- **Multi-Head Attention**: Multiple attention mechanisms in parallel for richer representations

### Transformer Architecture
- **Embeddings**: Token embeddings + positional embeddings
- **Transformer Blocks**: Multi-head attention + feedforward network with residual connections
- **Layer Normalization**: Stabilizes training (pre-norm architecture)
- **Output Head**: Projects to vocabulary for next-token prediction

### Training
- **Next-Token Prediction**: Model predicts next token given previous tokens
- **Cross-Entropy Loss**: Standard loss for classification/prediction tasks
- **Sliding Window**: Overlapping sequences maximize data utilization

### Fine-tuning
- **Transfer Learning**: Use pretrained model knowledge for downstream tasks
- **Classification Head**: Additional layer for class predictions
- **Freezing**: Keep pretrained weights fixed, train only new layers

## Model Configuration

The `GPT_CONFIG_124M` configuration includes:
- **vocab_size**: 50,257 (GPT-2 vocabulary)
- **emb_dim**: 768 (embedding dimension)
- **context_length**: 1024 (maximum sequence length)
- **n_layers**: 12 (number of transformer blocks)
- **n_heads**: 12 (attention heads per block)
- **drop_rate**: 0.1 (dropout probability)
- **qkv_bias**: False (no bias in Q, K, V projections)

## Dependencies

- PyTorch
- tiktoken (OpenAI BPE tokenizer)
- pandas (for CSV data loading)
- numpy

## File Organization

Each Python file is self-contained with:
- Comprehensive module-level documentation
- Detailed class and function docstrings
- Mathematical formulations where relevant
- Usage examples
- Concept explanations

## Learning Path

1. **Start with `tokenizer.py`**: Understand how text becomes numbers
2. **Study `attention.py`**: Learn the core attention mechanism
3. **Explore `transformer.py`**: See how components combine into a model
4. **Review `data_loader.py`**: Understand data preparation
5. **Practice with `training.py`**: Learn training and generation
6. **Experiment with `classification.py`**: Try fine-tuning

## Notes

- This is an educational implementation focused on clarity and understanding
- For production use, consider optimizations and additional features
- The code follows GPT-2 architecture principles
- All modules include extensive documentation for learning

## License

This is an educational project. Use and modify as needed for learning purposes.


