"""
Classification Fine-tuning Module

This module provides utilities for fine-tuning pretrained GPT models for
classification tasks, such as spam detection or sentiment analysis.

Key Components:
1. ClassificationHead: Linear layer for classification output
2. GPTForClassification: GPT model with classification head
3. Classification training and evaluation utilities

Key Concepts:
- Fine-tuning: Adapting a pretrained model to a specific task
- Classification Head: Additional layer that maps model outputs to class labels
- Freezing: Keeping pretrained weights fixed while training only new layers
- Transfer Learning: Using knowledge from pretraining for downstream tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import GPTModel
from typing import Optional


class ClassificationHead(nn.Module):
    """
    Classification head for fine-tuning GPT models.
    
    This module adds a classification layer on top of a pretrained GPT model.
    It takes the output from the last token position (or pooled representation)
    and projects it to the number of classes.
    
    Architecture:
        GPT Output -> LayerNorm -> Linear -> Class Logits
    
    The classification head is typically the only part trained during
    fine-tuning, while the GPT backbone can be frozen or fine-tuned.
    
    Attributes:
        norm (nn.LayerNorm): Layer normalization
        linear (nn.Linear): Classification layer
    
    Example:
        >>> head = ClassificationHead(emb_dim=768, num_classes=2)
        >>> gpt_output = torch.randn(2, 10, 768)  # [batch, seq_len, emb_dim]
        >>> logits = head(gpt_output)  # [2, 2]
    """
    
    def __init__(self, emb_dim: int, num_classes: int):
        """
        Initialize classification head.
        
        Args:
            emb_dim: Embedding dimension from GPT model
            num_classes: Number of classification classes
        """
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.linear = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Uses the last token's representation for classification.
        
        Args:
            x: GPT output of shape [batch_size, seq_len, emb_dim]
            
        Returns:
            Class logits of shape [batch_size, num_classes]
        """
        # Use last token representation
        x = x[:, -1, :]  # [batch_size, emb_dim]
        x = self.norm(x)
        logits = self.linear(x)
        return logits


class GPTForClassification(nn.Module):
    """
    GPT model with classification head for fine-tuning.
    
    This module combines a pretrained GPT model with a classification head,
    allowing the model to be fine-tuned for classification tasks like spam
    detection or sentiment analysis.
    
    Architecture:
        Input IDs -> GPT Model -> Classification Head -> Class Logits
    
    Training Strategy:
    - Option 1: Freeze GPT, train only classification head (faster, less memory)
    - Option 2: Fine-tune entire model (better performance, more resources)
    
    Attributes:
        gpt_model (GPTModel): Pretrained GPT model
        classification_head (ClassificationHead): Classification layer
        frozen (bool): Whether GPT model is frozen
    
    Example:
        >>> gpt_model = GPTModel(GPT_CONFIG_124M)
        >>> model = GPTForClassification(gpt_model, num_classes=2)
        >>> model.freeze_gpt()  # Freeze pretrained weights
        >>> logits = model(input_ids)  # [batch_size, 2]
    """
    
    def __init__(self, gpt_model: GPTModel, num_classes: int):
        """
        Initialize GPT model for classification.
        
        Args:
            gpt_model: Pretrained GPT model
            num_classes: Number of classification classes
        """
        super().__init__()
        self.gpt_model = gpt_model
        self.classification_head = ClassificationHead(
            emb_dim=gpt_model.tok_emb.embedding_dim,
            num_classes=num_classes
        )
        self.frozen = False
    
    def freeze_gpt(self):
        """
        Freeze the GPT model parameters.
        
        This prevents gradients from flowing through the GPT model during
        training, allowing only the classification head to be updated.
        Useful for:
        - Faster training
        - Lower memory usage
        - Preventing catastrophic forgetting
        """
        for param in self.gpt_model.parameters():
            param.requires_grad = False
        self.frozen = True
    
    def unfreeze_gpt(self):
        """
        Unfreeze the GPT model parameters.
        
        Allows fine-tuning of the entire model (GPT + classification head).
        This typically gives better performance but requires more resources.
        """
        for param in self.gpt_model.parameters():
            param.requires_grad = True
        self.frozen = False
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GPT and classification head.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            
        Returns:
            Class logits of shape [batch_size, num_classes]
        """
        # Get GPT outputs
        gpt_outputs = self.gpt_model(input_ids)  # [batch_size, seq_len, emb_dim]
        
        # Get logits from classification head
        logits = self.classification_head(gpt_outputs)
        return logits


def calculate_classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate cross-entropy loss for classification.
    
    Args:
        logits: Class logits of shape [batch_size, num_classes]
        labels: True class labels of shape [batch_size]
    
    Returns:
        Scalar loss tensor
    
    Example:
        >>> logits = model(input_ids)  # [2, 2]
        >>> labels = torch.tensor([0, 1])
        >>> loss = calculate_classification_loss(logits, labels)
    """
    return F.cross_entropy(logits, labels)


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        logits: Class logits of shape [batch_size, num_classes]
        labels: True class labels of shape [batch_size]
    
    Returns:
        Accuracy as a float between 0 and 1
    
    Example:
        >>> logits = model(input_ids)
        >>> accuracy = calculate_accuracy(logits, labels)
        >>> print(f"Accuracy: {accuracy * 100:.2f}%")
    """
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).float()
    return correct.mean().item()


def train_classification_epoch(model: GPTForClassification, dataloader,
                                optimizer: torch.optim.Optimizer,
                                device: str = "cpu") -> tuple:
    """
    Train classification model for one epoch.
    
    Args:
        model: GPTForClassification model
        dataloader: DataLoader providing (input_ids, labels) batches
        optimizer: Optimizer for training
        device: Device to run on
    
    Returns:
        Tuple of (average_loss, average_accuracy)
    
    Example:
        >>> optimizer = torch.optim.Adam(model.classification_head.parameters(), lr=1e-3)
        >>> loss, acc = train_classification_epoch(model, dataloader, optimizer)
    """
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(input_ids)
        loss = calculate_classification_loss(logits, labels)
        accuracy = calculate_accuracy(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
    return avg_loss, avg_accuracy


def evaluate_classification(model: GPTForClassification, dataloader,
                           device: str = "cpu") -> tuple:
    """
    Evaluate classification model on a dataset.
    
    Args:
        model: GPTForClassification model
        dataloader: DataLoader providing (input_ids, labels) batches
        device: Device to run on
    
    Returns:
        Tuple of (average_loss, average_accuracy)
    
    Example:
        >>> loss, acc = evaluate_classification(model, val_dataloader)
    """
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            logits = model(input_ids)
            loss = calculate_classification_loss(logits, labels)
            accuracy = calculate_accuracy(logits, labels)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
    return avg_loss, avg_accuracy


