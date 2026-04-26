from dataclasses import dataclass


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

