import torch
from torch import nn
from torch.nn import functional as F


class MLPLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(block_size * embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, time_steps = idx.shape
        if time_steps != self.block_size:
            raise ValueError(f"expected time dimension {self.block_size}, got {time_steps}")

        embeddings = self.token_embedding_table(idx)
        logits = self.network(embeddings.view(batch_size, -1))

        if targets is None:
            return logits, None

        next_targets = targets[:, -1]
        loss = F.cross_entropy(logits, next_targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            if idx_cond.shape[1] < self.block_size:
                pad = torch.zeros(
                    (idx_cond.shape[0], self.block_size - idx_cond.shape[1]),
                    dtype=idx_cond.dtype,
                    device=idx_cond.device,
                )
                idx_cond = torch.cat((pad, idx_cond), dim=1)

            logits, _ = self(idx_cond)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

