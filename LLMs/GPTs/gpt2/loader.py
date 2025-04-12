import os
import pickle
import torch
import tiktoken
from tqdm import tqdm
import numpy as np
import pandas as pd
import jieba
from collections import Counter

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
    

class DataLoaderChinese:
    def __init__(self, csv_file, batch_size, seq_length, cache_dir='./.cache'):
        """
        Initialize the DataLoader for a Chinese poetry dataset using jieba tokenizer.
        
        Args:
            csv_file (str): Path to the CSV file containing poetry data
            batch_size (int): Number of sequences in a batch (B)
            seq_length (int): Length of each sequence (T)
            cache_dir (str): Directory to store cached tokenized data
        """
        self.B = batch_size
        self.T = seq_length
        self.csv_file = csv_file
        self.cache_dir = cache_dir
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.EOS_TOKEN = '<EOS>'
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load the poetry data
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df)} poems from {csv_file}")
        
        # Tokenize the data using jieba with EOS tokens or load from cache
        self.vocab, self.tokens = self._tokenize_data()
        print(f"Vocabulary size: {len(self.vocab)}")
        
        # For simplicity, we'll use the entire token sequence as a single shard
        self.shards = [self.tokens]
        self.current_shard = 0
        
        # For compatibility with the original method (single process mode)
        self.process_rank = 0
        self.num_processes = 1
        
        # Set the initial position
        self.current_position = 0
    
    def _get_cache_path(self):
        """
        Generate a unique cache file path based on the input CSV file.
        
        Returns:
            tuple: (vocab_path, tokens_path) Paths to vocab and tokens cache files
        """
        # Create a filename based on the CSV filename
        base_name = os.path.basename(self.csv_file).split('.')[0].split("/")[-1]
        vocab_path = os.path.join(self.cache_dir, f"{base_name}_vocab.pkl")
        tokens_path = os.path.join(self.cache_dir, f"{base_name}_tokens.pt")
        return vocab_path, tokens_path
    
    def _tokenize_data(self):
        """
        Tokenize the poetry data using jieba and add EOS tokens.
        Caches results to disk and loads from cache if available.
        
        Returns:
            tuple: (vocab, tokens_tensor)
                vocab: Dictionary mapping tokens to indices
                tokens_tensor: Tensor of token indices
        """
        vocab_path, tokens_path = self._get_cache_path()
        
        # Check if cached data exists
        if os.path.exists(vocab_path) and os.path.exists(tokens_path):
            print(f"Loading tokenized data from cache...")
            # Load vocabulary and tokens from cache
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
            tokens = torch.load(tokens_path)
            return vocab, tokens
        
        print(f"Tokenizing data and creating cache...")
        # Process each poem individually to add EOS tokens
        all_tokens = []
        
        # Tokenize each poem with jieba and add EOS token
        for poem in tqdm(self.df.iloc[:, 0].astype(str), desc="tokenizing data:"):
            # Use jieba to segment the Chinese text
            poem_tokens = list(jieba.cut(poem, cut_all=False))
            # Add EOS token at the end of each poem
            poem_tokens.append(self.EOS_TOKEN)
            all_tokens.extend(poem_tokens)
        
        # Build vocabulary from tokens
        counter = Counter(all_tokens)
        sorted_vocab = sorted(counter.items(), key=lambda x: -x[1])
        
        # Create token to index mapping, ensuring special tokens have fixed indices
        vocab = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.EOS_TOKEN: 2,
        }
        
        # Add remaining tokens to vocabulary
        for idx, (token, _) in enumerate(sorted_vocab):
            if token not in vocab:  # Skip if already in special tokens
                vocab[token] = len(vocab)
        
        # Convert tokens to indices
        token_indices = []
        for token in all_tokens:
            token_indices.append(vocab[token])
        
        tokens = torch.tensor(token_indices, dtype=torch.long)
        
        # Save to cache
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        torch.save(tokens, tokens_path)
        
        print(f"Created and saved tokenized data to {self.cache_dir}")
        return vocab, tokens
    
    def next_batch(self):
        """
        Get the next batch of data.
        
        Returns:
            tuple: (x, y) where x is the input and y is the target
        """
        B, T = self.B, self.T
        
        # Check if we have enough tokens left
        if self.current_position + (B*T + 1) > len(self.tokens):
            # Wrap around to the beginning
            self.current_position = 0
        
        # Get tokens for the current batch
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        
        # Handle case where buffer is smaller than required
        if len(buf) < B*T + 1:
            # Pad with zeros or wrap around
            padding_needed = B*T + 1 - len(buf)
            buf = torch.cat([buf, self.tokens[:padding_needed]])
        
        # Reshape into input and target tensors
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)   # targets
        
        # Advance the position
        self.current_position += B * T * self.num_processes
        
        # If loading the next batch would be out of bounds, wrap around
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.shards[self.current_shard]
            self.current_position = B * T * self.process_rank
        
        return x, y