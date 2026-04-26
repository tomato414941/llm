import pytest

from llm.tokenizer import CharTokenizer


def test_char_tokenizer_round_trip() -> None:
    tokenizer = CharTokenizer.from_text("banana")

    encoded = tokenizer.encode("banana")

    assert tokenizer.decode(encoded) == "banana"
    assert tokenizer.vocab_size == 3


def test_char_tokenizer_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        CharTokenizer.from_text("")


def test_char_tokenizer_rejects_unknown_character() -> None:
    tokenizer = CharTokenizer.from_text("abc")

    with pytest.raises(ValueError, match="unknown character"):
        tokenizer.encode("abcd")


def test_char_tokenizer_rejects_unknown_token_id() -> None:
    tokenizer = CharTokenizer.from_text("abc")

    with pytest.raises(ValueError, match="unknown token id"):
        tokenizer.decode([99])

