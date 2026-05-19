"""
Attention Mechanisms Module for Transformer Architecture

This module implements various attention mechanisms used in transformer-based
language models, including self-attention, causal attention, and multi-head attention.

Key Concepts:
1. Self-Attention: Allows each position in a sequence to attend to all positions
   in the same sequence, computing a weighted sum of values based on query-key
   similarity.

2. Causal Attention: Self-attention with a causal mask that prevents positions
   from attending to future positions. Essential for autoregressive language models.

3. Multi-Head Attention: Runs multiple attention mechanisms in parallel, allowing
   the model to attend to information from different representation subspaces.

4. Scaled Dot-Product Attention: The core attention computation:
   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
   
   The scaling factor (sqrt(d_k)) prevents the dot products from growing too large,
   which would push the softmax into regions with extremely small gradients.

Mathematical Background:
- Query (Q): Represents "what am I looking for?"
- Key (K): Represents "what do I contain?"
- Value (V): Represents "what information do I provide?"
- Attention weights: Softmax of (Q @ K^T / sqrt(d_k))
- Context vector: Weighted sum of values using attention weights
"""

import torch
import torch.nn as nn


class SelfAttention_v1(nn.Module):
    """
    Basic self-attention mechanism using learnable parameter matrices.
    
    This is a simple implementation that uses nn.Parameter for the query, key,
    and value weight matrices. It's useful for understanding the core attention
    mechanism but SelfAttention_v2 is preferred for actual use.
    
    Attributes:
        W_query (nn.Parameter): Learnable query projection matrix [d_in, d_out]
        W_key (nn.Parameter): Learnable key projection matrix [d_in, d_out]
        W_value (nn.Parameter): Learnable value projection matrix [d_in, d_out]
    
    Example:
        >>> attn = SelfAttention_v1(d_in=512, d_out=256)
        >>> x = torch.randn(2, 10, 512)  # [batch, seq_len, d_in]
        >>> output = attn(x)  # [batch, seq_len, d_out]
    """
    
    def __init__(self, d_in: int, d_out: int):
        """
        Initialize self-attention layer.
        
        Args:
            d_in: Input embedding dimension
            d_out: Output embedding dimension
        """
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute self-attention.
        
        Args:
            x: Input tensor of shape [batch_size, num_tokens, d_in]
            
        Returns:
            Context vectors of shape [batch_size, num_tokens, d_out]
        """
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        # Compute attention scores: Q @ K^T
        attn_scores = queries @ keys.T
        
        # Scale and apply softmax to get attention weights
        # Scaling by sqrt(d_k) prevents large dot products that would
        # push softmax into regions with tiny gradients
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        # Weighted sum of values
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    """
    Improved self-attention using nn.Linear layers.
    
    This version uses PyTorch's nn.Linear layers instead of raw parameters,
    which provides better weight initialization and is the recommended approach.
    
    Key improvements over v1:
    - Uses nn.Linear for optimized weight initialization
    - Supports optional bias terms
    - More efficient and stable training
    
    Attributes:
        W_query (nn.Linear): Query projection layer
        W_key (nn.Linear): Key projection layer
        W_value (nn.Linear): Value projection layer
    
    Example:
        >>> attn = SelfAttention_v2(d_in=512, d_out=256, qkv_bias=False)
        >>> x = torch.randn(2, 10, 512)
        >>> output = attn(x)  # [2, 10, 256]
    """
    
    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        """
        Initialize self-attention layer.
        
        Args:
            d_in: Input embedding dimension
            d_out: Output embedding dimension
            qkv_bias: Whether to use bias in query, key, value projections
        """
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute self-attention.
        
        Args:
            x: Input tensor of shape [batch_size, num_tokens, d_in]
            
        Returns:
            Context vectors of shape [batch_size, num_tokens, d_out]
        """
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # Scaled dot-product attention
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    """
    Causal (masked) self-attention for autoregressive models.
    
    This attention mechanism prevents positions from attending to future positions
    by applying a causal mask. This is essential for language models that generate
    text autoregressively (one token at a time).
    
    Key Features:
    - Causal masking: Masks out future positions (upper triangular mask)
    - Dropout: Regularizes attention weights during training
    - Device-aware: Mask is automatically moved to the correct device
    
    The causal mask ensures that when computing attention for position i, only
    positions 0 to i are considered, preventing information leakage from
    future tokens.
    
    Attributes:
        W_query (nn.Linear): Query projection layer
        W_key (nn.Linear): Key projection layer
        W_value (nn.Linear): Value projection layer
        dropout (nn.Dropout): Dropout layer for attention weights
        mask (torch.Tensor): Causal attention mask (registered buffer)
    
    Example:
        >>> attn = CausalAttention(d_in=512, d_out=256, context_length=1024, dropout=0.1)
        >>> x = torch.randn(2, 10, 512)
        >>> output = attn(x)  # [2, 10, 256]
    """
    
    def __init__(self, d_in: int, d_out: int, context_length: int,
                 dropout: float, qkv_bias: bool = False):
        """
        Initialize causal attention layer.
        
        Args:
            d_in: Input embedding dimension
            d_out: Output embedding dimension
            context_length: Maximum sequence length (for mask creation)
            dropout: Dropout probability for attention weights
            qkv_bias: Whether to use bias in query, key, value projections
        """
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        
        # Register mask as buffer so it moves with the model to GPU/CPU
        # Upper triangular matrix: 1s above diagonal, 0s on/below diagonal
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute causal attention.
        
        Args:
            x: Input tensor of shape [batch_size, num_tokens, d_in]
            
        Returns:
            Context vectors of shape [batch_size, num_tokens, d_out]
        """
        b, num_tokens, d_in = x.shape
        
        # Project to query, key, value
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Compute attention scores
        attn_scores = queries @ keys.transpose(1, 2)
        
        # Apply causal mask: set future positions to -inf
        # This ensures positions can only attend to previous positions
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # Scale and apply softmax
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        
        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    """
    Multi-head attention implemented by stacking multiple CausalAttention modules.
    
    This is a straightforward but less efficient implementation. It creates
    multiple CausalAttention heads and concatenates their outputs.
    
    Note: This implementation processes heads sequentially. For better efficiency,
    use MultiHeadAttention which processes all heads in parallel.
    
    Attributes:
        heads (nn.ModuleList): List of CausalAttention modules
    
    Example:
        >>> mha = MultiHeadAttentionWrapper(
        ...     d_in=512, d_out=256, context_length=1024,
        ...     dropout=0.1, num_heads=8
        ... )
        >>> x = torch.randn(2, 10, 512)
        >>> output = mha(x)  # [2, 10, 256*8] = [2, 10, 2048]
    """
    
    def __init__(self, d_in: int, d_out: int, context_length: int,
                 dropout: float, num_heads: int, qkv_bias: bool = False):
        """
        Initialize multi-head attention wrapper.
        
        Args:
            d_in: Input embedding dimension
            d_out: Output dimension per head
            context_length: Maximum sequence length
            dropout: Dropout probability
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in projections
        """
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head attention by concatenating head outputs.
        
        Args:
            x: Input tensor of shape [batch_size, num_tokens, d_in]
            
        Returns:
            Concatenated context vectors of shape [batch_size, num_tokens, d_out*num_heads]
        """
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """
    Efficient multi-head attention with parallel head processing.
    
    This is the recommended implementation for multi-head attention. Instead of
    creating separate attention modules for each head, it splits a single large
    projection into multiple heads and processes them in parallel using efficient
    tensor operations.
    
    Key Advantages:
    - Parallel processing of all heads (more efficient)
    - Single matrix multiplication for Q, K, V projections
    - Output projection layer to combine head outputs
    - Better memory efficiency
    
    How it works:
    1. Project input to Q, K, V with dimension d_out (total across all heads)
    2. Reshape to split into num_heads separate heads
    3. Compute attention for all heads in parallel
    4. Concatenate head outputs
    5. Apply output projection
    
    Mathematical Formulation:
    - d_out must be divisible by num_heads
    - head_dim = d_out / num_heads
    - Each head processes with dimension head_dim
    - Final output has dimension d_out
    
    Attributes:
        W_query (nn.Linear): Query projection [d_in -> d_out]
        W_key (nn.Linear): Key projection [d_in -> d_out]
        W_value (nn.Linear): Value projection [d_in -> d_out]
        out_proj (nn.Linear): Output projection [d_out -> d_out]
        dropout (nn.Dropout): Dropout for attention weights
        mask (torch.Tensor): Causal attention mask
    
    Example:
        >>> mha = MultiHeadAttention(
        ...     d_in=512, d_out=512, context_length=1024,
        ...     dropout=0.1, num_heads=8
        ... )
        >>> x = torch.randn(2, 10, 512)
        >>> output = mha(x)  # [2, 10, 512]
    """
    
    def __init__(self, d_in: int, d_out: int, context_length: int,
                 dropout: float, num_heads: int, qkv_bias: bool = False):
        """
        Initialize multi-head attention layer.
        
        Args:
            d_in: Input embedding dimension
            d_out: Output embedding dimension (must be divisible by num_heads)
            context_length: Maximum sequence length
            dropout: Dropout probability for attention weights
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in Q, K, V projections
            
        Raises:
            AssertionError: If d_out is not divisible by num_heads
        """
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Dimension per head

        # Project to d_out dimension (will be split across heads)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # Output projection to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask (registered as buffer for device management)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head attention.
        
        Args:
            x: Input tensor of shape [batch_size, num_tokens, d_in]
            
        Returns:
            Context vectors of shape [batch_size, num_tokens, d_out]
        """
        b, num_tokens, d_in = x.shape

        # Project to Q, K, V: [b, num_tokens, d_out]
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape to split into heads: [b, num_tokens, num_heads, head_dim]
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose for parallel head processing: [b, num_heads, num_tokens, head_dim]
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores for all heads in parallel
        # [b, num_heads, num_tokens, num_tokens]
        attn_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # Scale and apply softmax
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        # Compute context vectors: [b, num_heads, num_tokens, head_dim]
        context_vec = attn_weights @ values
        
        # Transpose back: [b, num_tokens, num_heads, head_dim]
        context_vec = context_vec.transpose(1, 2)
        
        # Concatenate heads: [b, num_tokens, d_out]
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
        # Apply output projection
        context_vec = self.out_proj(context_vec)

        return context_vec


