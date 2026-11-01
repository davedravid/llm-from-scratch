"""Phase 6a: shared helpers for fine-tuning and preference tuning."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase4_transformer"))
sys.path.insert(0, str(ROOT / "phase5_generation"))

from generate import DEFAULT_CHECKPOINT as PRETRAINED, load_checkpoint, save_checkpoint  # noqa: E402,F401
from sampling import generate  # noqa: E402,F401
from tokens import bpe  # noqa: E402,F401

DATA = ROOT / "data"
CHECKPOINTS = Path(__file__).parent / "checkpoints"
IGNORE = -100   # cross_entropy skips targets with this value


def read_blocks(path, lines_per_block):
    blocks = [b.strip().split("\n") for b in path.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    bad = [b for b in blocks if len(b) != lines_per_block]
    if bad:
        raise ValueError(f"{path.name}: every block needs {lines_per_block} lines; first bad block: {bad[0]}")
    return blocks


def split_every(items, n):
    """Hold out every n-th item (spread through the file), return (train, held_out)."""
    return [x for i, x in enumerate(items) if i % n != n - 1], [x for i, x in enumerate(items) if i % n == n - 1]


def encode_pair(tokenizer, prompt, response, context_length):
    """Tokens for "prompt\\nresponse\\n", and targets where the PROMPT part is ignored.
    The final newline marks the end of the answer, so the model learns when to stop.

    The model sees the prompt, but it's only graded (loss) on writing the response. That's what makes
    this instruction tuning rather than more story training.
    """
    prompt_ids = tokenizer.encode(prompt)
    ids = tokenizer.encode(prompt + "\n" + response + "\n")
    if ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"prompt tokens changed when the response was added: {prompt!r}")
    if len(ids) > context_length + 1:
        raise ValueError(f"example is {len(ids)} tokens; the limit is {context_length + 1}: {prompt!r}")
    unknown = tokenizer.id_of[bpe.UNK]
    if unknown in ids:
        raise ValueError(f"example contains characters the tokenizer doesn't know: {prompt!r} / {response!r}")
    x, y = ids[:-1], ids[1:]
    y = [IGNORE] * (len(prompt_ids) - 1) + y[len(prompt_ids) - 1:]
    return x, y


def pad_batch(examples):
    """Stack examples of different lengths, padding the end. Padding is ignored by the loss, and the
    causal mask means it can't affect the real tokens before it."""
    width = max(len(x) for x, _ in examples)
    xs = torch.tensor([x + [0] * (width - len(x)) for x, _ in examples])
    ys = torch.tensor([y + [IGNORE] * (width - len(y)) for _, y in examples])
    return xs, ys


@torch.no_grad()
def response_loss(model, examples):
    """Average loss on the response tokens only."""
    was_training = model.training
    model.eval()
    xs, ys = pad_batch(examples)
    logits, _ = model(xs)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ys.reshape(-1), ignore_index=IGNORE)
    model.train(was_training)  # put it back as it was: training mode would switch dropout on
    return loss.item()


def response_logprob(model, examples):
    """Total log-probability the model gives each response (one number per example).
    Higher means the model considers that response more likely."""
    xs, ys = pad_batch(examples)
    logits, _ = model(xs)
    logp = torch.log_softmax(logits, dim=-1)
    mask = ys != IGNORE
    picked = logp.gather(-1, ys.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(dim=1)


def first_line(text):
    return text.strip().split("\n")[0]
