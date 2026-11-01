"""Phase 4a: a tiny GPT-style transformer, written by us.

What we use from PyTorch (external library):
  * tensors (multi-dimensional arrays) and automatic gradients
  * nn.Linear     - a weight matrix + bias: `x @ W + b`, exactly what Phase 2 did by hand
  * nn.Embedding  - a lookup table: row i is the vector for token i (like Phase 3b)
  * nn.LayerNorm, F.gelu, F.softmax, F.cross_entropy - standard maths helpers
What we write ourselves: attention (queries, keys, values, the causal mask, multiple heads),
the feed-forward layer, the block with residual connections, and the whole model.
We deliberately do NOT use PyTorch's ready-made nn.Transformer or nn.MultiheadAttention.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int            # how many different tokens (from the tokenizer)
    context_length: int = 32   # the most tokens the model can look at at once
    dim: int = 32              # numbers per token vector (the embedding size)
    heads: int = 4             # attention heads per block (dim must divide evenly by heads)
    layers: int = 2            # how many transformer blocks are stacked
    causal: bool = True        # the causal mask; only switched off to show why it's needed
    dropout: float = 0.0       # during training, randomly zero this fraction of signals (fights memorising)


class SelfAttention(nn.Module):
    """Every token looks at itself and the tokens before it, and mixes in what it finds."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.dim % cfg.heads == 0, "dim must be divisible by heads"
        self.cfg = cfg
        self.make_qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)  # learned: token vector -> q, k, v
        self.combine = nn.Linear(cfg.dim, cfg.dim)                   # learned: merge the heads back
        self.dropout = nn.Dropout(cfg.dropout)
        self.last_weights = None                                     # kept so we can inspect them

    def forward(self, x):
        batch, length, dim = x.shape
        heads, head_dim = self.cfg.heads, dim // self.cfg.heads

        # 1. Make a query, key and value for every token, then split them between the heads.
        q, k, v = self.make_qkv(x).split(dim, dim=2)
        q, k, v = (t.view(batch, length, heads, head_dim).transpose(1, 2) for t in (q, k, v))

        # 2. Score every query against every key (dot product), scaled so the numbers stay moderate.
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)       # (batch, heads, length, length)

        # 3. Causal mask: set scores for LATER tokens to -infinity, so they get weight 0.
        if self.cfg.causal:
            future = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(future, float("-inf"))

        # 4. Softmax turns each row of scores into weights that add up to 1.
        weights = F.softmax(scores, dim=-1)
        self.last_weights = weights.detach()

        # 5. Each token's output is the weighted mix of the values, one per head; then the heads are joined.
        out = (self.dropout(weights) @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.dropout(self.combine(out))


class FeedForward(nn.Module):
    """A small two-layer network applied to each token on its own (like Phase 2, just wider)."""

    def __init__(self, cfg):
        super().__init__()
        self.expand = nn.Linear(cfg.dim, 4 * cfg.dim)
        self.shrink = nn.Linear(4 * cfg.dim, cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.shrink(F.gelu(self.expand(x))))


class Block(nn.Module):
    """One transformer layer: attention (tokens share information), then feed-forward (each token thinks alone).

    `x + ...` is a residual connection: each step ADDS its result to the token vector instead of
    replacing it, so information (and training signal) can flow through many stacked blocks.
    LayerNorm rescales each vector to a standard size first, which keeps training stable.
    """

    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.attention = nn.LayerNorm(cfg.dim), SelfAttention(cfg)
        self.norm2, self.feed_forward = nn.LayerNorm(cfg.dim), FeedForward(cfg)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.dim)          # what each token is
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.dim)   # where it is in the sequence
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.final_norm = nn.LayerNorm(cfg.dim)
        self.to_logits = nn.Linear(cfg.dim, cfg.vocab_size)                   # one score per vocabulary token

    def forward(self, ids, targets=None):
        """ids: (batch, length) token IDs. Returns logits (batch, length, vocab_size) and the loss."""
        batch, length = ids.shape
        if length > self.cfg.context_length:
            raise ValueError(f"input is {length} tokens but the context window is {self.cfg.context_length}")
        if ids.min() < 0 or ids.max() >= self.cfg.vocab_size:
            raise ValueError(f"token IDs must be between 0 and {self.cfg.vocab_size - 1}")

        positions = torch.arange(length, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)    # token meaning + position
        for block in self.blocks:
            x = block(x)
        logits = self.to_logits(self.final_norm(x))

        # At every position the target is the NEXT token. Cross-entropy is low when the model
        # gave the correct next token a high probability.
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, ids, new_tokens):
        """Autoregressive generation: predict one token, append it, repeat.

        This is greedy decoding: always take the single most likely next token. Phase 5 adds
        temperature and top-p sampling. Only the last `context_length` tokens are fed in each
        time, because that's all the model can see.
        """
        for _ in range(new_tokens):
            window = ids[:, -self.cfg.context_length:]
            logits, _ = self(window)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids
