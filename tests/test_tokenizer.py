import pytest

from llm.tokenizer import BPETokenizer, CharTokenizer


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


def test_bpe_tokenizer_round_trip() -> None:
    tokenizer = BPETokenizer.train("banana bandana", vocab_size=270)

    encoded = tokenizer.encode("banana bandana")

    assert tokenizer.decode(encoded) == "banana bandana"
    assert tokenizer.vocab_size > 256


def test_bpe_tokenizer_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        BPETokenizer.train("", vocab_size=256)


def test_bpe_tokenizer_rejects_too_small_vocab() -> None:
    with pytest.raises(ValueError, match="at least 256"):
        BPETokenizer.train("abc", vocab_size=255)


def test_bpe_tokenizer_save_load_round_trip(tmp_path) -> None:
    path = tmp_path / "tokenizer.json"
    tokenizer = BPETokenizer.train("hello hello", vocab_size=265)

    tokenizer.save(path)
    loaded = BPETokenizer.load(path)

    assert loaded.decode(loaded.encode("hello hello")) == "hello hello"
    assert loaded.vocab == tokenizer.vocab
    assert loaded.merges == tokenizer.merges


def test_bpe_tokenizer_rejects_unknown_token_id() -> None:
    tokenizer = BPETokenizer.train("abc", vocab_size=256)

    with pytest.raises(ValueError, match="unknown token id"):
        tokenizer.decode([999])
