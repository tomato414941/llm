import torch
from torch import nn
from torch.nn import functional as F


class SelfAttentionHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        head_size: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time_steps, channels = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * channels**-0.5
        weights = weights.masked_fill(self.tril[:time_steps, :time_steps] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        head_size: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    embedding_dim=embedding_dim,
                    head_size=head_size,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )
        self.projection = nn.Linear(num_heads * head_size, embedding_dim)
        self.projection_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.projection_dropout(self.projection(out))
