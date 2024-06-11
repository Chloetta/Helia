import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import nbimporter
from llama.model import Transformer  # Import Transformer class from llama.model
from llama.tokenizer import Tokenizer  # Import Tokenizer class from llama.tokenizer
import requests
from bs4 import BeautifulSoup
import time
import json
from dataclasses import dataclass
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

# Load TensorTorch Module from a Jupyter notebook
TensorTorch = nbimporter.import_notebook('Torch.ipynb')
from TensorTorch import LinearLayer, MultiHeadAttention, FeedForward

# Defined Parameters using dataclass
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer to power of two
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 32
    max_seq_len: int = 2048

# Define a simple GELU activation function
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# Define a simple attention mask function
def attention_mask(nd, ns, dtype=torch.float32):
    i = torch.arange(nd)[:, None]
    j = torch.arange(ns)
    m = i >= j - ns + nd
    return m.to(dtype)

# Define single multi-headed block
class AttentionBlock(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super(AttentionBlock, self).__init__()
        self.model_args = model_args
        self.multi_head_attn = MultiHeadAttention(model_args.dim, model_args.n_heads)
        self.ffn = FeedForward(model_args.dim, model_args.dim * 4)

    def forward(self, x, mask):
        attn_output = self.multi_head_attn(x, x, x, mask)
        ffn_output = self.ffn(attn_output)
        return ffn_output

# Example usage
if __name__ == "__main__":
    args = ModelArgs(vocab_size=50257)
    # Example tensor input
    x = torch.rand((args.max_batch_size, args.max_seq_len, args.dim))
    mask = attention_mask(args.max_seq_len, args.max_seq_len)

    # Initialize and run attention block
    attn_block = AttentionBlock(args)
    output = attn_block(x, mask)
    print(output.shape)
