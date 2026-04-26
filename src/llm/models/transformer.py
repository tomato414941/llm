from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from llm.models.attention import MultiHeadAttention


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    block_size: int
    embedding_dim: int
    num_heads: int
    num_layers: int

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "TransformerConfig":
        return cls(**data)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, block_size: int) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            head_size=embedding_dim // num_heads,
            block_size=block_size,
        )
        self.feed_forward = FeedForward(embedding_dim)
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding_table = nn.Embedding(config.block_size, config.embedding_dim)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    block_size=config.block_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size)

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
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        _, _, channels = logits.shape
        loss = F.cross_entropy(logits.view(batch_size * time_steps, channels), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
