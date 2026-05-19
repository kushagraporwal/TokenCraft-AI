"""
Transformer Architecture Module for GPT-style Language Models

This module implements the complete transformer architecture used in GPT models,
including layer normalization, feedforward networks, transformer blocks, and
the full GPT model.

Key Components:
1. GELU: Gaussian Error Linear Unit activation function
2. LayerNorm: Layer normalization for stabilizing training
3. FeedForward: Two-layer MLP with GELU activation
4. TransformerBlock: Core transformer block with attention and feedforward
5. GPTModel: Complete GPT model architecture

Architecture Overview:
- Token embeddings + Positional embeddings
- Stack of TransformerBlocks (with multi-head attention and feedforward)
- Final layer normalization
- Output head (linear projection to vocabulary)
"""

import torch
import torch.nn as nn
from attention import MultiHeadAttention


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation function.
    
    GELU is a smooth, non-linear activation function that approximates ReLU
    but with non-zero gradients for negative values. This makes it particularly
    useful for transformer architectures as it allows for more nuanced parameter
    adjustments during training.
    
    Mathematical Formulation:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    
    Key Properties:
    - Smooth (differentiable everywhere)
    - Non-zero output for negative inputs (unlike ReLU)
    - Better optimization properties in deep networks
    
    Example:
        >>> gelu = GELU()
        >>> x = torch.tensor([-1.0, 0.0, 1.0])
        >>> output = gelu(x)
    """
    
    def __init__(self):
        """Initialize GELU activation function."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            GELU-activated tensor
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class LayerNorm(nn.Module):
    """
    Layer Normalization module.
    
    Layer normalization normalizes inputs across the features (embedding dimension)
    for each sample independently. This helps stabilize training and allows for
    better gradient flow in deep networks.
    
    Mathematical Formulation:
        LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β
        
    where:
        - μ is the mean across the feature dimension
        - σ² is the variance across the feature dimension
        - γ (scale) and β (shift) are learnable parameters
        - ε is a small constant to prevent division by zero
    
    In GPT models, layer normalization is applied before attention and feedforward
    layers (pre-norm architecture), which has better training dynamics than
    post-norm architectures.
    
    Attributes:
        scale (nn.Parameter): Learnable scale parameter γ
        shift (nn.Parameter): Learnable shift parameter β
        eps (float): Small constant for numerical stability
    
    Example:
        >>> norm = LayerNorm(emb_dim=768)
        >>> x = torch.randn(2, 10, 768)  # [batch, seq_len, emb_dim]
        >>> normalized = norm(x)  # Same shape
    """
    
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        """
        Initialize layer normalization.
        
        Args:
            emb_dim: Embedding dimension (feature dimension to normalize over)
            eps: Small constant for numerical stability (default: 1e-5)
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Normalizes across the last dimension (embedding dimension) for each
        sample independently.
        
        Args:
            x: Input tensor of shape [batch_size, num_tokens, emb_dim]
            
        Returns:
            Normalized tensor of same shape
        """
        # Compute mean and variance across the embedding dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable scale and shift
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    """
    Feedforward Neural Network module for transformer blocks.
    
    This is a two-layer MLP that expands the embedding dimension, applies
    non-linear activation, then contracts back to the original dimension.
    
    Architecture:
        Input (emb_dim) -> Linear (4 * emb_dim) -> GELU -> Linear (emb_dim) -> Output
    
    The expansion to 4x the embedding dimension allows the model to explore
    a richer representation space before contracting back. This design is
    standard in transformer architectures.
    
    Attributes:
        layers (nn.Sequential): Sequential layers: Linear -> GELU -> Linear
    
    Example:
        >>> cfg = {"emb_dim": 768}
        >>> ffn = FeedForward(cfg)
        >>> x = torch.randn(2, 10, 768)
        >>> output = ffn(x)  # [2, 10, 768]
    """
    
    def __init__(self, cfg: dict):
        """
        Initialize feedforward network.
        
        Args:
            cfg: Configuration dictionary containing "emb_dim"
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # Expand
            GELU(),                                          # Non-linearity
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # Contract
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feedforward network.
        
        Args:
            x: Input tensor of shape [batch_size, num_tokens, emb_dim]
            
        Returns:
            Output tensor of same shape [batch_size, num_tokens, emb_dim]
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    Transformer Block: Core building block of GPT models.
    
    A transformer block consists of:
    1. Multi-head self-attention (with causal masking)
    2. Feedforward neural network
    3. Layer normalization (applied before each sub-layer - pre-norm)
    4. Residual connections (shortcut connections)
    5. Dropout for regularization
    
    Architecture Flow:
        x -> LayerNorm -> Attention -> Dropout -> + (residual) -> 
        LayerNorm -> FeedForward -> Dropout -> + (residual) -> output
    
    The residual connections help with gradient flow in deep networks and
    allow the model to learn identity mappings when needed.
    
    Pre-LayerNorm vs Post-LayerNorm:
    - This implementation uses Pre-LayerNorm (normalize before sub-layer)
    - Pre-norm generally has better training dynamics than post-norm
    - Post-norm was used in the original Transformer paper
    
    Attributes:
        att (MultiHeadAttention): Multi-head attention module
        ff (FeedForward): Feedforward network module
        norm1 (LayerNorm): Layer norm before attention
        norm2 (LayerNorm): Layer norm before feedforward
        drop_shortcut (nn.Dropout): Dropout for residual connections
    
    Example:
        >>> cfg = {
        ...     "emb_dim": 768,
        ...     "context_length": 1024,
        ...     "n_heads": 12,
        ...     "drop_rate": 0.1,
        ...     "qkv_bias": False
        ... }
        >>> block = TransformerBlock(cfg)
        >>> x = torch.randn(2, 10, 768)
        >>> output = block(x)  # [2, 10, 768]
    """
    
    def __init__(self, cfg: dict):
        """
        Initialize transformer block.
        
        Args:
            cfg: Configuration dictionary containing:
                - emb_dim: Embedding dimension
                - context_length: Maximum sequence length
                - n_heads: Number of attention heads
                - drop_rate: Dropout probability
                - qkv_bias: Whether to use bias in Q, K, V projections
        """
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape [batch_size, num_tokens, emb_dim]
            
        Returns:
            Output tensor of same shape [batch_size, num_tokens, emb_dim]
        """
        # Attention block with residual connection
        shortcut = x
        x = self.norm1(x)  # Pre-norm
        x = self.att(x)    # Multi-head attention
        x = self.drop_shortcut(x)
        x = x + shortcut   # Residual connection

        # Feedforward block with residual connection
        shortcut = x
        x = self.norm2(x)  # Pre-norm
        x = self.ff(x)     # Feedforward network
        x = self.drop_shortcut(x)
        x = x + shortcut   # Residual connection

        return x


class GPTModel(nn.Module):
    """
    Complete GPT (Generative Pre-trained Transformer) Model.
    
    This is a decoder-only transformer architecture used for autoregressive
    language modeling. The model predicts the next token in a sequence given
    all previous tokens.
    
    Architecture:
        1. Token Embeddings: Convert token IDs to dense vectors
        2. Positional Embeddings: Add positional information
        3. Dropout: Regularize embeddings
        4. Transformer Blocks: Stack of N transformer blocks
        5. Final Layer Norm: Normalize before output
        6. Output Head: Linear projection to vocabulary size (logits)
    
    Forward Pass:
        token_ids -> embeddings -> + positional -> dropout ->
        transformer_blocks -> layer_norm -> output_head -> logits
    
    The output logits represent unnormalized probabilities over the vocabulary.
    To get probabilities, apply softmax. To get the next token, use argmax or
    sample from the distribution.
    
    Attributes:
        tok_emb (nn.Embedding): Token embedding layer
        pos_emb (nn.Embedding): Positional embedding layer
        drop_emb (nn.Dropout): Dropout for embeddings
        trf_blocks (nn.Sequential): Stack of transformer blocks
        final_norm (LayerNorm): Final layer normalization
        out_head (nn.Linear): Output projection to vocabulary
    
    Example:
        >>> cfg = {
        ...     "vocab_size": 50257,
        ...     "emb_dim": 768,
        ...     "context_length": 1024,
        ...     "n_layers": 12,
        ...     "n_heads": 12,
        ...     "drop_rate": 0.1,
        ...     "qkv_bias": False
        ... }
        >>> model = GPTModel(cfg)
        >>> token_ids = torch.randint(0, 50257, (2, 10))  # [batch, seq_len]
        >>> logits = model(token_ids)  # [2, 10, 50257]
    """
    
    def __init__(self, cfg: dict):
        """
        Initialize GPT model.
        
        Args:
            cfg: Configuration dictionary containing:
                - vocab_size: Size of vocabulary
                - emb_dim: Embedding dimension
                - context_length: Maximum sequence length
                - n_layers: Number of transformer blocks
                - n_heads: Number of attention heads
                - drop_rate: Dropout probability
                - qkv_bias: Whether to use bias in Q, K, V projections
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Stack of transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        # Final normalization and output head
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GPT model.
        
        Args:
            in_idx: Token IDs tensor of shape [batch_size, seq_len]
            
        Returns:
            Logits tensor of shape [batch_size, seq_len, vocab_size]
            Each position contains logits for predicting the next token
        """
        batch_size, seq_len = in_idx.shape
        
        # Token embeddings
        tok_embeds = self.tok_emb(in_idx)
        
        # Positional embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        # Combine embeddings
        x = tok_embeds + pos_embeds  # [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        
        # Pass through transformer blocks
        x = self.trf_blocks(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Output logits
        logits = self.out_head(x)
        return logits


