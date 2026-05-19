"""
Tokenizer Module for LLM Training

This module provides tokenization utilities for converting text into token IDs
that can be used for training Large Language Models (LLMs).

The module includes:
1. SimpleTokenizerV1: Basic tokenizer with vocabulary mapping
2. SimpleTokenizerV2: Enhanced tokenizer with unknown token handling
3. BPETokenizer: Wrapper for tiktoken BPE tokenizer (GPT-2 style)

Key Concepts:
- Tokenization: The process of breaking text into smaller units (tokens)
- Vocabulary: A mapping between tokens and their integer IDs
- BPE (Byte Pair Encoding): A subword tokenization algorithm that can handle
  out-of-vocabulary words by breaking them into subword units
"""

import re
import tiktoken
from typing import List, Dict, Union


class SimpleTokenizerV1:
    """
    A simple tokenizer that maps text tokens to integer IDs.
    
    This tokenizer splits text on whitespace and punctuation, then maps each
    token to a unique integer ID based on a provided vocabulary.
    
    Attributes:
        str_to_int (Dict[str, int]): Mapping from token strings to integer IDs
        int_to_str (Dict[int, str]): Inverse mapping from integer IDs to token strings
    
    Example:
        >>> vocab = {"hello": 0, "world": 1, ",": 2}
        >>> tokenizer = SimpleTokenizerV1(vocab)
        >>> ids = tokenizer.encode("hello, world")
        >>> text = tokenizer.decode(ids)
    """
    
    def __init__(self, vocab: Dict[str, int]):
        """
        Initialize the tokenizer with a vocabulary.
        
        Args:
            vocab: Dictionary mapping token strings to integer IDs
        """
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text into a list of token IDs.
        
        The text is split on whitespace and punctuation characters. Each token
        is then looked up in the vocabulary to get its corresponding ID.
        
        Args:
            text: Input text string to tokenize
            
        Returns:
            List of integer token IDs
            
        Note:
            If a token is not in the vocabulary, this will raise a KeyError.
            Use SimpleTokenizerV2 for unknown token handling.
        """
        # Split on whitespace and punctuation
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        
        # Remove whitespace and empty strings
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        # Map tokens to IDs
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids: List[int]) -> str:
        """
        Convert a list of token IDs back into text.
        
        Args:
            ids: List of integer token IDs
            
        Returns:
            Reconstructed text string
            
        Note:
            Spaces are automatically removed before punctuation marks for
            proper text formatting.
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    """
    Enhanced tokenizer with unknown token handling.
    
    This tokenizer extends SimpleTokenizerV1 by handling out-of-vocabulary
    (OOV) words using a special <|unk|> token. It also supports the <|endoftext|>
    special token for marking document boundaries.
    
    Attributes:
        str_to_int (Dict[str, int]): Mapping from token strings to integer IDs
        int_to_str (Dict[int, str]): Inverse mapping from integer IDs to token strings
    
    Example:
        >>> vocab = {"hello": 0, "world": 1, "<|unk|>": 2, "<|endoftext|>": 3}
        >>> tokenizer = SimpleTokenizerV2(vocab)
        >>> ids = tokenizer.encode("hello unknownword")
        >>> # Unknown words are replaced with <|unk|> token
    """
    
    def __init__(self, vocab: Dict[str, int]):
        """
        Initialize the tokenizer with a vocabulary.
        
        Args:
            vocab: Dictionary mapping token strings to integer IDs.
                   Should include "<|unk|>" and "<|endoftext|>" tokens.
        """
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text into a list of token IDs, handling unknown words.
        
        Unknown words (not in vocabulary) are replaced with the <|unk|> token.
        This allows the tokenizer to process any text, even with words not seen
        during vocabulary creation.
        
        Args:
            text: Input text string to tokenize
            
        Returns:
            List of integer token IDs
        """
        # Split on whitespace and punctuation
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        # Replace unknown tokens with <|unk|>
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids: List[int]) -> str:
        """
        Convert a list of token IDs back into text.
        
        Args:
            ids: List of integer token IDs
            
        Returns:
            Reconstructed text string with proper punctuation spacing
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer wrapper using tiktoken.
    
    BPE is a subword tokenization algorithm that breaks words into smaller
    subword units. This allows the tokenizer to handle any word, even if it
    wasn't seen during training, by representing it as a combination of
    subword tokens.
    
    This tokenizer uses the GPT-2 BPE implementation from OpenAI's tiktoken
    library, which was used to train GPT-2, GPT-3, and the original ChatGPT model.
    
    Key Features:
    - Vocabulary size: 50,257 tokens (for GPT-2 encoding)
    - Handles out-of-vocabulary words automatically
    - Uses <|endoftext|> token (ID: 50256) for document boundaries
    - No <|unk|> token needed due to subword encoding
    
    Attributes:
        encoding: The tiktoken encoding object
        vocab_size: Size of the vocabulary
        
    Example:
        >>> tokenizer = BPETokenizer()
        >>> ids = tokenizer.encode("Hello, world!")
        >>> text = tokenizer.decode(ids)
    """
    
    def __init__(self, encoding_name: str = "gpt2"):
        """
        Initialize the BPE tokenizer.
        
        Args:
            encoding_name: Name of the tiktoken encoding to use.
                          Options: "gpt2", "p50k_base" (GPT-3), "cl100k_base" (GPT-4)
                          Default: "gpt2"
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab
    
    def encode(self, text: str, allowed_special: set = None) -> List[int]:
        """
        Convert text into a list of token IDs using BPE.
        
        The BPE algorithm breaks down words into subword units, allowing
        any word to be tokenized even if it wasn't in the training vocabulary.
        
        Args:
            text: Input text string to tokenize
            allowed_special: Set of special tokens to allow (e.g., {"<|endoftext|>"})
                           If None, special tokens raise errors
            
        Returns:
            List of integer token IDs
            
        Example:
            >>> tokenizer = BPETokenizer()
            >>> ids = tokenizer.encode("Hello unknownword!", allowed_special=set())
            >>> # Even "unknownword" gets tokenized into subword units
        """
        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}
        return self.encoding.encode(text, allowed_special=allowed_special)
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert a list of token IDs back into text.
        
        Args:
            ids: List of integer token IDs
            
        Returns:
            Reconstructed text string
        """
        return self.encoding.decode(ids)
    
    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size of the tokenizer.
        
        Returns:
            Number of tokens in the vocabulary
        """
        return self.vocab_size


def create_vocab_from_text(text: str, add_special_tokens: bool = True) -> Dict[str, int]:
    """
    Create a vocabulary dictionary from text.
    
    This function extracts all unique tokens from the text and creates a
    vocabulary mapping. Useful for creating vocabularies for SimpleTokenizer
    classes.
    
    Args:
        text: Input text to extract vocabulary from
        add_special_tokens: If True, adds <|endoftext|> and <|unk|> tokens
        
    Returns:
        Dictionary mapping token strings to integer IDs
        
    Example:
        >>> text = "Hello world. Hello again."
        >>> vocab = create_vocab_from_text(text)
        >>> # vocab contains: {"Hello": 0, "world": 1, ".": 2, "again": 3, ...}
    """
    # Split text into tokens
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    
    # Get unique tokens and sort
    all_tokens = sorted(list(set(preprocessed)))
    
    # Add special tokens if requested
    if add_special_tokens:
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    
    # Create vocabulary mapping
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    return vocab


