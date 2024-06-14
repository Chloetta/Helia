import math
from typing import Optional
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def attention_mask(nd, ns, dtype=torch.float32):
    i = torch.arange(nd)[:, None]
    j = torch.arange(ns)
    m = i >= j - ns + nd
    return m.to(dtype)

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.query_linear = nn.Linear(dim, dim)
        self.key_linear = nn.Linear(dim, dim)
        self.value_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)
        attention = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim)
        if mask is not None:
            attention = attention + mask
        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, value)
        return output

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super(AttentionBlock, self).__init__()
        self.model_args = model_args
        self.multi_head_attn = Attention(model_args.dim, model_args.n_heads)
        self.ffn = FeedForward(model_args.dim, model_args.dim * 4)

    def forward(self, x, mask):
        attn_output = self.multi_head_attn(x, mask)
        ffn_output = self.ffn(attn_output)
        return ffn_output

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output

class VocabParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, init_method):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.init_method = init_method

    def forward(self, x):
        return self.embedding(x)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return x / (x.norm(dim=-1, keepdim=True) + self.eps)

class ColumnParallelLinear(nn.Module):
    def __init__(self, dim, vocab_size, bias=False):
        super().__init__()
        self.linear = nn.Linear(dim, vocab_size, bias=bias)

    def forward(self, x):
        return self.linear(x)

def precompute_freqs_cis(dim, max_seq_len, rope_theta):
    freqs_cis = torch.zeros((max_seq_len, dim))
    for i in range(max_seq_len):
        freqs_cis[i] = torch.arange(i, i + dim) / (rope_theta * math.sqrt(dim))
    return freqs_cis

class TransformerBlock(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.args = args
        self.ln1 = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.attn = Attention(args.dim, args.n_heads)
        self.ln2 = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.ffn = FeedForward(args.dim, args.dim * 4)

    def forward(self, x, start_pos, freqs_cis, mask):
        h = self.ln1(x)
        attn_output = self.attn(h, mask)
        h = x + attn_output
        h = self.ln2(h)
        ffn_output = self.ffn(h)
        output = x + ffn_output
        return output