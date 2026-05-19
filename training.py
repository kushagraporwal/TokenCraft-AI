"""
Training Utilities Module for LLM Pretraining

This module provides utilities for training GPT-style language models,
including loss computation, text generation, and training loop helpers.

Key Functions:
1. calculate_loss: Compute cross-entropy loss for next-token prediction
2. generate_text: Generate text using the trained model
3. Training loop utilities and helpers

Key Concepts:
- Next-Token Prediction: The model predicts the next token given previous tokens
- Cross-Entropy Loss: Standard loss function for classification tasks
- Text Generation: Autoregressive generation where each token depends on previous tokens
- Temperature Sampling: Controls randomness in text generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import tiktoken


def calculate_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate cross-entropy loss for next-token prediction.
    
    This function computes the standard cross-entropy loss used for training
    language models. The logits represent unnormalized probabilities over the
    vocabulary, and targets are the true next token IDs.
    
    Mathematical Formulation:
        Loss = -log(P(target_token | context))
    
    The loss is computed for each position in the sequence and then averaged.
    
    Args:
        logits: Model output of shape [batch_size, seq_len, vocab_size]
                Unnormalized log probabilities over vocabulary
        targets: Target token IDs of shape [batch_size, seq_len]
                 True next tokens for each position
    
    Returns:
        Scalar tensor representing the average loss
    
    Example:
        >>> logits = model(input_ids)  # [2, 10, 50257]
        >>> loss = calculate_loss(logits, target_ids)  # scalar
    """
    # Reshape for cross-entropy: [batch*seq_len, vocab_size] and [batch*seq_len]
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(batch_size * seq_len, vocab_size)
    targets_flat = targets.view(batch_size * seq_len)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(logits_flat, targets_flat)
    return loss


def generate_text(model: nn.Module, tokenizer, context: str, max_new_tokens: int = 50,
                  temperature: float = 1.0, top_k: Optional[int] = None,
                  device: str = "cpu") -> str:
    """
    Generate text using a trained GPT model.
    
    This function performs autoregressive text generation, where each new token
    is predicted based on all previous tokens. The generation continues until
    max_new_tokens is reached or an end-of-text token is generated.
    
    Generation Strategies:
    1. Greedy: Always pick the most likely token (temperature=0)
    2. Sampling: Sample from the probability distribution (temperature>0)
    3. Top-k Sampling: Sample from top-k most likely tokens
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer with encode() and decode() methods
        context: Starting text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (0=greedy, >0=sampling)
                    Higher temperature = more random
        top_k: If specified, only sample from top-k tokens
        device: Device to run generation on ("cpu" or "cuda")
    
    Returns:
        Generated text string (context + generated tokens)
    
    Example:
        >>> model = GPTModel(GPT_CONFIG_124M)
        >>> tokenizer = tiktoken.get_encoding("gpt2")
        >>> text = generate_text(model, tokenizer, "The future of AI is")
    """
    model.eval()
    
    # Encode context
    context_ids = tokenizer.encode(context, allowed_special={"<|endoftext|>"})
    context_tensor = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    generated_ids = context_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits for the last position
            logits = model(context_tensor)  # [1, seq_len, vocab_size]
            logits = logits[0, -1, :] / temperature  # Last position, [vocab_size]
            
            # Apply top-k filtering if specified
            if top_k is not None:
                # Get top-k values and indices
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                # Create a new tensor with -inf everywhere except top-k
                filtered_logits = torch.full_like(logits, float('-inf'))
                filtered_logits[top_k_indices] = top_k_logits
                logits = filtered_logits
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            if temperature == 0:
                # Greedy: pick most likely token
                next_token = torch.argmax(probs).item()
            else:
                # Sample from distribution
                next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Check for end-of-text token
            if next_token == tokenizer.encode("<|endoftext|>")[0]:
                break
            
            # Append to generated sequence
            generated_ids.append(next_token)
            
            # Update context for next iteration
            context_tensor = torch.cat([
                context_tensor,
                torch.tensor([[next_token]], dtype=torch.long, device=device)
            ], dim=1)
            
            # Truncate context if it exceeds max length
            max_context = model.pos_emb.num_embeddings
            if context_tensor.shape[1] > max_context:
                context_tensor = context_tensor[:, -max_context:]
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids)
    return generated_text


def train_epoch(model: nn.Module, dataloader, optimizer: torch.optim.Optimizer,
                device: str = "cpu") -> float:
    """
    Train the model for one epoch.
    
    This function runs a single training epoch, processing all batches in the
    dataloader and updating model parameters.
    
    Args:
        model: GPT model to train
        dataloader: DataLoader providing (input_ids, target_ids) batches
        optimizer: Optimizer for updating model parameters
        device: Device to run training on ("cpu" or "cuda")
    
    Returns:
        Average loss over the epoch
    
    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        >>> loss = train_epoch(model, dataloader, optimizer, device="cuda")
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        # Move to device
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Forward pass
        logits = model(input_ids)
        loss = calculate_loss(logits, target_ids)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model: nn.Module, dataloader, device: str = "cpu") -> float:
    """
    Evaluate the model on a validation/test set.
    
    This function computes the average loss on a dataset without updating
    model parameters (no gradient computation).
    
    Args:
        model: GPT model to evaluate
        dataloader: DataLoader providing (input_ids, target_ids) batches
        device: Device to run evaluation on ("cpu" or "cuda")
    
    Returns:
        Average loss on the dataset
    
    Example:
        >>> val_loss = evaluate(model, val_dataloader, device="cuda")
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            loss = calculate_loss(logits, target_ids)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


