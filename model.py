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

# Load Torch Module via Jupyter notebook
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

class Embeddings(nn.Module):
    def __init__(self, vocab_size, dim):
        super(Embeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, dim)
        self.position_embeddings = nn.Embedding(5000, dim)  # assuming max sequence length of 5000

    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0).to(x.device)
        return self.token_embeddings(x) + self.position_embeddings(positions)

class TransformerModel(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super(TransformerModel, self).__init__()
        self.embeddings = Embeddings(model_args.vocab_size, model_args.dim)
        self.blocks = nn.ModuleList([AttentionBlock(model_args) for _ in range(model_args.n_layers)])
        self.norm = nn.LayerNorm(model_args.dim, eps=model_args.norm_eps)
        self.output_layer = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def forward(self, x, mask):
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.output_layer(x)

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids, labels = batch
        input_ids, labels = input_ids.to(device), labels.to(device)
        mask = attention_mask(input_ids.size(1), input_ids.size(1)).to(device)
        outputs = model(input_ids, mask)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)
            mask = attention_mask(input_ids.size(1), input_ids.size(1)).to(device)
            outputs = model(input_ids, mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()