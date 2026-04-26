from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CharTokenizer:
    chars: tuple[str, ...]
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("text must not be empty")
        chars = tuple(sorted(set(text)))
        stoi = {char: index for index, char in enumerate(chars)}
        itos = {index: char for index, char in enumerate(chars)}
        return cls(chars=chars, stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[char] for char in text]
        except KeyError as error:
            raise ValueError(f"unknown character: {error.args[0]!r}") from error

    def decode(self, token_ids: list[int]) -> str:
        try:
            return "".join(self.itos[token_id] for token_id in token_ids)
        except KeyError as error:
            raise ValueError(f"unknown token id: {error.args[0]!r}") from error


@dataclass(frozen=True)
class BPETokenizer:
    vocab: dict[int, bytes]
    merges: tuple[tuple[int, int], ...]

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        if not text:
            raise ValueError("text must not be empty")
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        tokens = list(text.encode("utf-8"))
        vocab = {index: bytes([index]) for index in range(256)}
        merges = []

        for token_id in range(256, vocab_size):
            stats = cls._get_pair_counts(tokens)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            tokens = cls._merge(tokens, pair, token_id)
            vocab[token_id] = vocab[pair[0]] + vocab[pair[1]]
            merges.append(pair)

        return cls(vocab=vocab, merges=tuple(merges))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        for index, pair in enumerate(self.merges, start=256):
            tokens = self._merge(tokens, pair, index)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        try:
            raw = b"".join(self.vocab[token_id] for token_id in token_ids)
        except KeyError as error:
            raise ValueError(f"unknown token id: {error.args[0]!r}") from error
        return raw.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab": {str(token_id): token_bytes.hex() for token_id, token_bytes in self.vocab.items()},
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in payload["vocab"].items()}
        merges = tuple((int(left), int(right)) for left, right in payload["merges"])
        return cls(vocab=vocab, merges=merges)

    @staticmethod
    def _get_pair_counts(tokens: list[int]) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(tokens: list[int], pair: tuple[int, int], new_token_id: int) -> list[int]:
        new_tokens = []
        index = 0
        while index < len(tokens):
            if index < len(tokens) - 1 and (tokens[index], tokens[index + 1]) == pair:
                new_tokens.append(new_token_id)
                index += 2
            else:
                new_tokens.append(tokens[index])
                index += 1
        return new_tokens
