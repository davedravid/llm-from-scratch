"""Phase 5a: turning the model's scores into the next token, written by us.

    logits (raw scores) --divide by temperature--> softmax --> probabilities
        --top-p keeps the most likely tokens--> pick one at random by probability

Also: the context window, i.e. what happens when the text is longer than the model can see.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase4_transformer"))

from generate import DEFAULT_CHECKPOINT, load_checkpoint  # noqa: E402,F401  (Phase 4 code)
from tokens import bpe  # noqa: E402,F401


def check_settings(temperature, top_p):
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be more than 0 and at most 1")


def softmax_with_temperature(logits, temperature):
    """Divide the scores by the temperature, then softmax.

    Temperature below 1 stretches the gaps between scores (the favourite gets even more likely);
    above 1 squashes them (unlikely tokens get a bigger share). Temperature 0 would mean dividing
    by zero, so it's handled as a special case: all probability on the single top token.
    """
    if temperature == 0:
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
        return probs
    return torch.softmax(logits / temperature, dim=-1)


def top_p_filter(probs, top_p):
    """Keep the SMALLEST set of most-likely tokens whose probabilities add up to at least top_p,
    zero the rest, and rescale so the kept ones add up to 1 again.

    Not "the top p% of tokens": if one token has 95% probability, top_p=0.9 keeps just that one;
    if 200 tokens each have 0.5%, it keeps 180 of them.
    """
    sorted_probs, order = probs.sort(descending=True)
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs   # probability of the tokens ranked above
    keep = mass_before < top_p                                  # the top token is always kept
    filtered = torch.zeros_like(probs)
    filtered[order[keep]] = sorted_probs[keep]
    return filtered / filtered.sum()


def next_token_probs(logits, temperature=1.0, top_p=1.0):
    check_settings(temperature, top_p)
    return top_p_filter(softmax_with_temperature(logits, temperature), top_p)


def sample_next(logits, temperature=1.0, top_p=1.0, generator=None):
    """Pick the next token ID from one position's logits."""
    probs = next_token_probs(logits, temperature, top_p)
    return int(torch.multinomial(probs, 1, generator=generator))


def fit_to_context(ids, context_length):
    """Return (kept, dropped): only the LAST `context_length` tokens fit; older ones fall off the front."""
    return ids[-context_length:], ids[:-context_length] if len(ids) > context_length else ids[:0]


@torch.no_grad()
def generate(model, tokenizer, prompt, new_tokens, temperature=1.0, top_p=1.0, seed=None):
    """The autoregressive loop with sampling: predict, pick, append, repeat."""
    check_settings(temperature, top_p)
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("the prompt is empty")
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    start = len(ids)
    for _ in range(new_tokens):
        window, _ = fit_to_context(ids, model.cfg.context_length)
        logits, _ = model(torch.tensor([window]))
        ids.append(sample_next(logits[0, -1], temperature, top_p, generator))
    return tokenizer.decode(ids[start:])
