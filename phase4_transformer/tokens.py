"""Gives the transformer access to the Phase 3a BPE tokenizer."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase3a_tokenizer"))

import bpe  # noqa: E402  (our tokenizer from Phase 3a)


def load_tokenizer():
    """Load the saved tokenizer, training one with the Phase 3a defaults if it doesn't exist yet."""
    if not bpe.DEFAULT_SAVE.exists():
        text = bpe.DEFAULT_DATA.read_text(encoding="utf-8")
        bpe.train(text, num_merges=500).save(bpe.DEFAULT_SAVE)
    return bpe.Tokenizer.load(bpe.DEFAULT_SAVE)
