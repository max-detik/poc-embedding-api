"""Token-budgeted chunking.

Pure over a tokenizer: works with any Hugging Face tokenizer, or anything
with the same __call__ / decode / num_special_tokens_to_add surface.
"""

from typing import List


def chunk_text(
    tokenizer,
    text: str,
    max_seq_length: int,
    reserved_text: str = "",
    chunk_overlap: int = 0,
) -> List[str]:
    """Split `text` into pieces that still fit `max_seq_length` once
    `reserved_text` (template prefix, title, separators) is added around each.

    Returns [text] unchanged when it already fits.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    reserved_len = (
        len(tokenizer(reserved_text, add_special_tokens=False)["input_ids"])
        if reserved_text else 0
    )
    special_len = max(tokenizer.num_special_tokens_to_add(pair=False), 1)
    budget = max_seq_length - reserved_len - special_len
    if budget <= 0:
        raise ValueError(
            f"max_seq_length={max_seq_length} too small for the document "
            f"template plus title ({reserved_len} tokens)"
        )
    if len(ids) <= budget:
        return [text]
    if chunk_overlap >= budget:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk size ({budget})")

    step = budget - chunk_overlap
    chunks = []
    for start in range(0, len(ids), step):
        chunks.append(tokenizer.decode(ids[start : start + budget], skip_special_tokens=True))
        if start + budget >= len(ids):
            break
    return chunks
