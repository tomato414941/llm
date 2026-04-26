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


class SingleHeadAttentionLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.head = SelfAttentionHead(
            embedding_dim=embedding_dim,
            head_size=embedding_dim,
            block_size=block_size,
        )
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, time_steps = idx.shape
        if time_steps > self.block_size:
            raise ValueError(f"expected time dimension <= {self.block_size}, got {time_steps}")

        token_embeddings = self.token_embedding_table(idx)
        position_ids = torch.arange(time_steps, device=idx.device)
        position_embeddings = self.position_embedding_table(position_ids)
        x = token_embeddings + position_embeddings
        x = self.head(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        _, _, channels = logits.shape
        loss = F.cross_entropy(logits.view(batch_size * time_steps, channels), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class MultiHeadAttentionLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embedding_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            head_size=embedding_dim // num_heads,
            block_size=block_size,
        )
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, time_steps = idx.shape
        if time_steps > self.block_size:
            raise ValueError(f"expected time dimension <= {self.block_size}, got {time_steps}")

        token_embeddings = self.token_embedding_table(idx)
        position_ids = torch.arange(time_steps, device=idx.device)
        position_embeddings = self.position_embedding_table(position_ids)
        x = token_embeddings + position_embeddings
        x = self.attention(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        _, _, channels = logits.shape
        loss = F.cross_entropy(logits.view(batch_size * time_steps, channels), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
