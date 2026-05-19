"""
Data Loading Module for LLM Training and Fine-tuning

This module provides PyTorch Dataset classes and DataLoader utilities for
loading and preprocessing text data for language model training and fine-tuning.

Key Components:
1. GPTDatasetV1: Dataset for pretraining (next-token prediction)
2. SpamDataset: Dataset for classification fine-tuning
3. InstructionDataset: Dataset for instruction fine-tuning
4. create_dataloader_v1: Utility function to create data loaders

Key Concepts:
- Sliding Window: For pretraining, text is split into overlapping sequences
  using a sliding window approach. This maximizes data utilization.
- Input-Target Pairs: For next-token prediction, targets are inputs shifted
  by one position (causal language modeling).
- Padding: For classification tasks, sequences are padded to the same length
  for batching.
- Tokenization: Text is converted to token IDs using BPE tokenizer.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import pandas as pd
from typing import Optional, List, Tuple, Callable


class GPTDatasetV1(Dataset):
    """
    Dataset for GPT pretraining (next-token prediction).
    
    This dataset creates input-target pairs from text using a sliding window
    approach. For each position in the text, it creates a sequence of tokens
    (input) and the corresponding sequence shifted by one position (target).
    
    Sliding Window Approach:
    - Text is tokenized into a sequence of token IDs
    - A sliding window of size `max_length` moves through the text
    - Step size is controlled by `stride`
    - Smaller stride = more overlapping sequences = more training data
    
    Input-Target Relationship:
    - Input: tokens[i:i+max_length]
    - Target: tokens[i+1:i+max_length+1]
    - This creates next-token prediction tasks
    
    Example:
        Text: "Hello world"
        Tokenized: [15496, 995]  (assuming max_length=2, stride=1)
        Input: [15496, 995]
        Target: [995, <next_token>]
    
    Attributes:
        input_ids (List[torch.Tensor]): List of input token sequences
        target_ids (List[torch.Tensor]): List of target token sequences
    
    Example:
        >>> tokenizer = tiktoken.get_encoding("gpt2")
        >>> dataset = GPTDatasetV1(text, tokenizer, max_length=256, stride=128)
        >>> input_ids, target_ids = dataset[0]
    """
    
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        """
        Initialize GPT dataset.
        
        Args:
            txt: Input text string to create dataset from
            tokenizer: Tokenizer object with encode() method (e.g., tiktoken)
            max_length: Maximum sequence length (context window size)
            stride: Step size for sliding window (smaller = more overlap)
        """
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use sliding window to create overlapping sequences
        for i in range(0, len(token_ids) - max_length, stride):
            # Input sequence: tokens from position i to i+max_length
            input_chunk = token_ids[i:i + max_length]
            
            # Target sequence: input shifted by 1 position (next-token prediction)
            target_chunk = token_ids[i + 1: i + max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """
        Get the number of sequences in the dataset.
        
        Returns:
            Number of input-target pairs
        """
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single input-target pair.
        
        Args:
            idx: Index of the sequence to retrieve
            
        Returns:
            Tuple of (input_ids, target_ids) tensors
        """
        return self.input_ids[idx], self.target_ids[idx]


class SpamDataset(Dataset):
    """
    Dataset for spam classification fine-tuning.
    
    This dataset loads text-label pairs from a CSV file and prepares them
    for classification. Text is tokenized and sequences are padded/truncated
    to a fixed length for batching.
    
    Data Format:
    - CSV file should have "Text" and "Label" columns
    - Labels are typically 0 (not spam) or 1 (spam)
    
    Preprocessing:
    1. Tokenize all texts
    2. Truncate sequences longer than max_length
    3. Pad sequences shorter than max_length to max_length
    4. Return token IDs and labels as tensors
    
    Attributes:
        data (pd.DataFrame): Loaded CSV data
        encoded_texts (List[List[int]]): Pre-tokenized texts
        max_length (int): Maximum sequence length
    
    Example:
        >>> tokenizer = tiktoken.get_encoding("gpt2")
        >>> dataset = SpamDataset("spam_data.csv", tokenizer, max_length=512)
        >>> token_ids, label = dataset[0]
    """
    
    def __init__(self, csv_file: str, tokenizer, max_length: Optional[int] = None,
                 pad_token_id: int = 50256):
        """
        Initialize spam classification dataset.
        
        Args:
            csv_file: Path to CSV file with "Text" and "Label" columns
            tokenizer: Tokenizer object with encode() method
            max_length: Maximum sequence length. If None, uses longest sequence
            pad_token_id: Token ID to use for padding (default: 50256 = <|endoftext|>)
        """
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize all texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        # Determine max_length
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            
            # Truncate sequences longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to max_length
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def _longest_encoded_length(self) -> int:
        """
        Find the length of the longest encoded sequence.
        
        Returns:
            Length of longest sequence
        """
        return max(len(encoded) for encoded in self.encoded_texts)

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of text-label pairs
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single text-label pair.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Tuple of (token_ids, label) where:
            - token_ids: Tensor of shape [max_length]
            - label: Integer label (0 or 1)
        """
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )


class InstructionDataset(Dataset):
    """
    Dataset for instruction fine-tuning.
    
    This dataset loads instruction-response pairs and formats them for
    fine-tuning language models to follow instructions. The format typically
    includes a prompt/instruction and a response.
    
    The dataset applies a formatting function to each instruction-response
    pair before tokenization, allowing for flexible prompt templates.
    
    Attributes:
        data (pd.DataFrame): Loaded instruction data
        tokenizer: Tokenizer for encoding text
        format_input (Callable): Function to format instruction-response pairs
        encoded_texts (List[List[int]]): Pre-tokenized formatted texts
    
    Example:
        >>> def format_func(instruction, response):
        ...     return f"Instruction: {instruction}\\nResponse: {response}"
        >>> dataset = InstructionDataset(
        ...     "instructions.csv", tokenizer, format_func
        ... )
    """
    
    def __init__(self, csv_file: str, tokenizer, format_input: Callable,
                 max_length: Optional[int] = None):
        """
        Initialize instruction dataset.
        
        Args:
            csv_file: Path to CSV file with instruction data
            tokenizer: Tokenizer object with encode() method
            format_input: Function that takes (instruction, response) and
                         returns formatted string
            max_length: Maximum sequence length (None = no limit)
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.format_input = format_input
        
        # Format and tokenize all instruction-response pairs
        self.encoded_texts = []
        for idx in range(len(self.data)):
            formatted = format_input(
                self.data.iloc[idx]["Instruction"],
                self.data.iloc[idx]["Response"]
            )
            encoded = tokenizer.encode(formatted, allowed_special={"<|endoftext|>"})
            
            # Truncate if max_length is specified
            if max_length is not None:
                encoded = encoded[:max_length]
            
            self.encoded_texts.append(encoded)

    def __len__(self) -> int:
        """
        Get the number of instruction-response pairs.
        
        Returns:
            Number of samples
        """
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Get a single formatted and tokenized instruction-response pair.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Tensor of token IDs
        """
        return torch.tensor(self.encoded_texts[index], dtype=torch.long)


def create_dataloader_v1(txt: str, batch_size: int = 4, max_length: int = 256,
                         stride: int = 128, shuffle: bool = True,
                         drop_last: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Create a DataLoader for GPT pretraining.
    
    This is a convenience function that creates a GPTDatasetV1 and wraps it
    in a PyTorch DataLoader for efficient batch processing during training.
    
    Parameters:
    - batch_size: Number of sequences per batch
    - max_length: Context window size (sequence length)
    - stride: Step size for sliding window
    - shuffle: Whether to shuffle data each epoch
    - drop_last: Whether to drop last incomplete batch
    - num_workers: Number of worker processes for data loading
    
    Stride Guidelines:
    - stride = max_length: No overlap between sequences (efficient, less data)
    - stride < max_length: Overlapping sequences (more data, potential overfitting)
    - Common: stride = max_length / 2 (50% overlap)
    
    Example:
        >>> with open("book.txt", "r") as f:
        ...     text = f.read()
        >>> dataloader = create_dataloader_v1(
        ...     text, batch_size=8, max_length=256, stride=128
        ... )
        >>> for inputs, targets in dataloader:
        ...     # inputs: [batch_size, max_length]
        ...     # targets: [batch_size, max_length]
        ...     pass
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


