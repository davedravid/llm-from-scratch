"""Phase 5b: application-level guardrails around the Phase 4b model.

The model only continues text. Everything here is ordinary code in the APPLICATION, running
before the model sees the input and after it produces output. These checks work no matter how
the model behaves, which is exactly why they can't live inside a prompt.

Our pretend product: a children's story generator. Its policy (stand-in, harmless examples):
no violent or insulting words, no instruction-override attempts, no reciting training text.
"""
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase5_generation"))

import metrics  # noqa: E402  (Phase 5a)
from sampling import DEFAULT_CHECKPOINT, generate, load_checkpoint  # noqa: E402

# ---------------------------------------------------------------------------
# Part 1: the policy, as data. Every threshold is a product decision, not a fact.
# ---------------------------------------------------------------------------

MAX_INPUT_CHARS = 500
MAX_UNKNOWN_SHARE = 0.2        # block if more than 20% of tokens are characters never seen in training
MIN_FAMILIAR_SHARE = 0.5       # flag if fewer than half the words appear in the training stories
MAX_COPY_WORDS = 12            # block output that copies 12+ words in a row from training text
MIN_OUTPUT_REAL_WORDS = 0.5    # flag output where fewer than half the words are real words
BLOCKED_WORDS = {"kill", "hurt", "weapon", "stupid"}
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above)\s+(instructions|rules)",
    r"system\s+prompt",
    r"you\s+are\s+now",
    r"developer\s+mode",
]


@dataclass
class Decision:
    status: str                                   # allowed | flagged | blocked_input | blocked_output
    text: str = ""                                # what the user gets back (empty if blocked)
    reasons: list = field(default_factory=list)   # why it was blocked or flagged
    raw_output: str = ""                          # what the model actually wrote (kept for the log)


# ---------------------------------------------------------------------------
# Part 2: the checks. Each returns (block_reasons, flag_reasons).
# ---------------------------------------------------------------------------

def check_input(prompt, tokenizer, known_words, context_length, blocked_words=BLOCKED_WORDS, check_familiar=True):
    if not isinstance(prompt, str):
        return [f"input is not text (got {type(prompt).__name__})"], []
    if not prompt.strip():
        return ["input is empty"], []
    if len(prompt) > MAX_INPUT_CHARS:
        return [f"input is too long ({len(prompt)} characters, limit {MAX_INPUT_CHARS})"], []

    block, flag = [], []
    if any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in prompt):
        block.append("input contains control characters")
    words = metrics.words(prompt)
    if hits := sorted(blocked_words & set(words)):
        block.append(f"input contains blocked words: {', '.join(hits)}")
    if any(re.search(p, prompt.lower()) for p in INJECTION_PATTERNS):
        block.append("input looks like an attempt to override instructions")

    ids = tokenizer.encode(prompt)
    unknown = ids.count(tokenizer.id_of["<unk>"]) / len(ids)
    if unknown > MAX_UNKNOWN_SHARE:
        block.append(f"{unknown:.0%} of the input is characters the model has never seen")
    elif check_familiar and metrics.real_word_rate(prompt, known_words) < MIN_FAMILIAR_SHARE:
        flag.append("unfamiliar input: most words never appear in the training stories, output will be unreliable")
    if len(ids) > context_length:
        flag.append(f"input is {len(ids)} tokens; only the last {context_length} will be used")
    return block, flag


def check_output(text, training_text, known_words, blocked_words=BLOCKED_WORDS):
    block, flag = [], []
    if hits := sorted(blocked_words & set(metrics.words(text))):
        block.append(f"output contains blocked words: {', '.join(hits)}")
    if (copied := metrics.longest_copy(text, training_text)) >= MAX_COPY_WORDS:
        block.append(f"output copies {copied} words in a row from the training data")
    if not metrics.words(text):
        flag.append("output contains no words")
    elif metrics.real_word_rate(text, known_words) < MIN_OUTPUT_REAL_WORDS:
        flag.append("low-quality output: fewer than half the words are real")
    return block, flag


# ---------------------------------------------------------------------------
# Part 3: the guarded pipeline: check input -> model -> check output -> log.
# ---------------------------------------------------------------------------

class GuardedModel:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, blocked_words=BLOCKED_WORDS):
        self.model, self.tokenizer, info = load_checkpoint(checkpoint)
        self.training_text = info["training_text"]
        self.known_words = set(metrics.words(self.training_text))
        self.blocked_words = set(blocked_words)
        self.log = []   # every decision, as a production system would record for monitoring

    def respond(self, prompt, guards=True, temperature=0.8, top_p=0.9, seed=0, tokens=40):
        decision = self._respond(prompt, guards, temperature, top_p, seed, tokens)
        self.log.append({"prompt": prompt if isinstance(prompt, str) else repr(prompt), "guards": guards,
                         "status": decision.status, "reasons": decision.reasons})
        return decision

    def _respond(self, prompt, guards, temperature, top_p, seed, tokens):
        flags = []
        if guards:
            block, flags = check_input(prompt, self.tokenizer, self.known_words,
                                       self.model.cfg.context_length, self.blocked_words)
            if block:
                return Decision("blocked_input", reasons=block)
        try:
            output = generate(self.model, self.tokenizer, prompt, tokens, temperature, top_p, seed)
        except (ValueError, TypeError, AttributeError) as error:   # without guards, bad input reaches the model
            return Decision("error", reasons=[f"model failed: {error}"])
        if not guards:
            return Decision("allowed", text=output, raw_output=output)
        block, out_flags = check_output(output, self.training_text, self.known_words, self.blocked_words)
        if block:
            return Decision("blocked_output", reasons=block, raw_output=output)
        flags += out_flags
        return Decision("flagged" if flags else "allowed", text=output, reasons=flags, raw_output=output)
