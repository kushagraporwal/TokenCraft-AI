"""
Model Configuration Module

This module defines configuration dictionaries for different GPT model sizes.
These configurations specify the hyperparameters and architecture details
for training GPT-style language models.

Key Hyperparameters:
- vocab_size: Size of the tokenizer vocabulary (50,257 for GPT-2)
- emb_dim: Embedding dimension (same for token and positional embeddings)
- context_length: Maximum sequence length the model can process
- n_layers: Number of transformer blocks
- n_heads: Number of attention heads
- drop_rate: Dropout probability
- qkv_bias: Whether to use bias in query, key, value projections

Model Sizes:
- GPT_CONFIG_124M: 124 million parameters (smallest GPT-2 size)
- Additional configurations can be added for larger models
"""

# GPT-2 Small Configuration (124M parameters)
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # GPT-2 vocabulary size
    "emb_dim": 768,           # Embedding dimension
    "context_length": 1024,   # Maximum sequence length (inference default)
    "n_layers": 12,           # Number of transformer blocks
    "n_heads": 12,            # Number of attention heads per block
    "drop_rate": 0.1,         # Dropout probability
    "qkv_bias": False         # No bias in Q, K, V projections
}

# Training config for small corpora (matches LLM Pretraining.ipynb)
GPT_CONFIG_124M_TRAIN = {
    **GPT_CONFIG_124M,
    "context_length": 256,
}

# Additional configurations can be added here:
# GPT_CONFIG_350M = {...}
# GPT_CONFIG_774M = {...}
# GPT_CONFIG_1558M = {...}

def get_model_size(cfg: dict) -> int:
    """
    Estimate the number of parameters in a GPT model configuration.
    
    This is a rough estimate. The actual parameter count may vary slightly
    due to implementation details.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Estimated number of parameters
    """
    vocab_size = cfg["vocab_size"]
    emb_dim = cfg["emb_dim"]
    n_layers = cfg["n_layers"]
    n_heads = cfg["n_heads"]
    context_length = cfg["context_length"]
    
    # Token embeddings
    tok_emb = vocab_size * emb_dim
    
    # Positional embeddings
    pos_emb = context_length * emb_dim
    
    # Per transformer block:
    # - Multi-head attention: 4 * emb_dim^2 (Q, K, V, output projection)
    # - Feedforward: 2 * emb_dim * (4 * emb_dim) = 8 * emb_dim^2
    # - Layer norms: 2 * emb_dim (scale and shift per norm)
    params_per_block = 12 * emb_dim * emb_dim + 2 * emb_dim
    
    # All transformer blocks
    trf_blocks = n_layers * params_per_block
    
    # Final layer norm
    final_norm = 2 * emb_dim
    
    # Output head (no bias)
    out_head = emb_dim * vocab_size
    
    total = tok_emb + pos_emb + trf_blocks + final_norm + out_head
    return total


