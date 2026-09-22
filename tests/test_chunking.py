import pytest

from embedder.chunking import chunk_text

WORDS = " ".join(f"w{i}" for i in range(250))


def test_short_text_is_returned_unchanged(tokenizer):
    assert chunk_text(tokenizer, "pendek saja", max_seq_length=100) == ["pendek saja"]


def test_chunks_fit_budget_after_reserved_text(tokenizer):
    reserved = "Judul artikel yang cukup panjang\n\n"  # 5 tokens
    chunks = chunk_text(tokenizer, WORDS, max_seq_length=60, reserved_text=reserved)
    for c in chunks:
        full = tokenizer(reserved + c)["input_ids"]  # includes the special token
        assert len(full) <= 60
    assert chunks[-1].split()[-1] == "w249"  # nothing dropped at the end


def test_overlap_is_carried_between_chunks(tokenizer):
    chunks = chunk_text(tokenizer, WORDS, max_seq_length=100, chunk_overlap=10)
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-10:] == b.split()[:10]


def test_reserved_text_too_long(tokenizer):
    with pytest.raises(ValueError, match="too small"):
        chunk_text(tokenizer, WORDS, max_seq_length=5, reserved_text="a b c d e f")


def test_overlap_must_be_smaller_than_chunk(tokenizer):
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text(tokenizer, WORDS, max_seq_length=20, chunk_overlap=50)
