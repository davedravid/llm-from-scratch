"""Phase 4b: training data for the transformer.

The stories are split into a TRAINING set (the model learns from these) and a
VALIDATION set (held back; only used to measure how well the model does on
text it has never seen). The tokenizer is also trained on the training
stories only, so nothing about the validation stories leaks into the model.
"""
import torch

from tokens import bpe

STORIES_FILE = bpe.DEFAULT_DATA
SEPARATOR = "\n\n"


def load_stories(path=STORIES_FILE):
    text = path.read_text(encoding="utf-8")
    return [story.strip() for story in text.split(SEPARATOR) if story.strip()]


def split_stories(stories, val_count):
    """Hold out `val_count` stories, spread evenly through the file (not just the last few)."""
    if not 1 <= val_count < len(stories):
        raise ValueError(f"val_count must be between 1 and {len(stories) - 1}")
    step = len(stories) / val_count
    val_idx = {int(i * step + step / 2) for i in range(val_count)}
    train = [s for i, s in enumerate(stories) if i not in val_idx]
    val = [s for i, s in enumerate(stories) if i in val_idx]
    return train, val


def get_batch(ids, batch_size, length, generator):
    """Pick `batch_size` random windows of `length` tokens. The target for each position is the
    next token, so targets are the same window shifted one place to the right."""
    if len(ids) <= length:
        raise ValueError(f"need more than {length} tokens of data, got {len(ids)}")
    starts = torch.randint(0, len(ids) - length, (batch_size,), generator=generator)
    x = torch.stack([ids[s:s + length] for s in starts])
    y = torch.stack([ids[s + 1:s + length + 1] for s in starts])
    return x, y
