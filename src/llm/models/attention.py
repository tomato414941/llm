import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = x.shape
        q, k, v = self.qkv(x).split(self.embedding_dim, dim=-1)
        q = q.view(batch_size, time_steps, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, time_steps, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, time_steps, self.num_heads, self.head_size).transpose(1, 2)

        weights = q @ k.transpose(-2, -1) * self.head_size**-0.5
        weights = weights.masked_fill(self.causal_mask[:, :, :time_steps, :time_steps] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.attention_dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, time_steps, self.embedding_dim)
        return self.projection_dropout(self.projection(out))
