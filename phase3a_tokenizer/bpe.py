"""Phase 3a: a byte pair encoding (BPE) tokenizer, written from scratch.

BPE learns a vocabulary from text: start with single characters, then
repeatedly glue together the pair of neighbouring pieces that appears most
often. Common words end up as one token; rare words stay split into pieces.

Run (from the project folder):
    .venv\\Scripts\\python phase3a_tokenizer\\bpe.py
    .venv\\Scripts\\python phase3a_tokenizer\\bpe.py --merges 20 --text "Once upon a time"
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_DATA = HERE.parent / "data" / "stories.txt"
DEFAULT_SAVE = HERE / "tokenizer.json"
UNK = "<unk>"  # stands in for any character never seen in training

# ---------------------------------------------------------------------------
# Part 1: split text into chunks before BPE. A chunk is a word or a run of
# punctuation, with any spaces in front of it attached:
#   "Once upon a time." -> ["Once", " upon", " a", " time", "."]
# Merges only happen inside a chunk, so a token never spans two words.
# (That's our design choice, copied from GPT-style tokenizers; it isn't part of BPE itself.)
# ---------------------------------------------------------------------------

CHUNK_PATTERN = re.compile(r"\s*\w+|\s*[^\w\s]+|\s+")


def split_into_chunks(text):
    return CHUNK_PATTERN.findall(text)


def merge_pair(symbols, pair):
    """Replace every neighbouring (a, b) in `symbols` with the single piece a+b."""
    merged, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


# ---------------------------------------------------------------------------
# Part 2: training = learning the list of merges.
# ---------------------------------------------------------------------------

def train(text, num_merges, min_count=2, show=0):
    # Count each distinct chunk once, with how often it occurs: " the" may appear 300 times,
    # but we only store it once with a count of 300 (the speed-up the video describes).
    chunk_counts = Counter(tuple(chunk) for chunk in split_into_chunks(text))
    merges = []
    for step in range(num_merges):
        pair_counts = Counter()
        for symbols, count in chunk_counts.items():
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += count
        if not pair_counts:
            break
        # Most frequent pair wins; ties are broken alphabetically so every run gives the same result.
        best, count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
        if count < min_count:
            break  # merging a pair seen only once just memorises that one word
        merges.append(best)
        chunk_counts = Counter({merge_pair(s, best): c for s, c in chunk_counts.items()})
        if step < show:
            print(f"  merge {step + 1:>3}: {show_token(best[0])} + {show_token(best[1])}"
                  f" -> {show_token(best[0] + best[1])}   (seen {count} times)")

    # Vocabulary = <unk>, every single character, then each merged piece in the order learned.
    vocab = [UNK] + sorted(set(text))
    for a, b in merges:
        if a + b not in vocab:
            vocab.append(a + b)
    return Tokenizer(vocab, merges)


# ---------------------------------------------------------------------------
# Part 3: using the learned merges on any new text.
# ---------------------------------------------------------------------------

class Tokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab                                  # id -> token text
        self.merges = [tuple(m) for m in merges]
        self.rank = {pair: i for i, pair in enumerate(self.merges)}  # earlier merge = higher priority
        self.id_of = {token: i for i, token in enumerate(vocab)}      # token text -> id

    def tokenize(self, text):
        """Text -> list of token strings, applying merges in the order they were learned."""
        tokens = []
        for chunk in split_into_chunks(text):
            symbols = tuple(chunk)
            while len(symbols) > 1:
                pairs = zip(symbols, symbols[1:])
                best = min(pairs, key=lambda p: self.rank.get(p, float("inf")))
                if best not in self.rank:
                    break  # no learned merge applies any more
                symbols = merge_pair(symbols, best)
            tokens.extend(symbols)
        return tokens

    def encode(self, text):
        """Text -> list of token IDs. This is what a model actually receives."""
        unk = self.id_of[UNK]
        return [self.id_of.get(token, unk) for token in self.tokenize(text)]

    def decode(self, ids):
        """Token IDs -> text."""
        return "".join(self.vocab[i] for i in ids)

    def save(self, path):
        path.write_text(json.dumps({"vocab": self.vocab, "merges": self.merges}, indent=1))

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text())
        return cls(data["vocab"], data["merges"])


# ---------------------------------------------------------------------------
# Part 4: the demo.
# ---------------------------------------------------------------------------

DEMO_TEXTS = [
    "Once upon a time there was a king.",            # looks like the training data
    "The astronaut zoomed past Jupiter.",            # words never seen in training
    "Il etait une fois un roi.",                     # French: same alphabet, different language
    "The cat 🙂 sat.",                               # a character never seen in training
]


def show_token(token):
    """Make spaces and newlines visible: ' king' -> '_king'."""
    return token.replace(" ", "_").replace("\n", "\\n")


def describe(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    print(f"  text   : {text!r}")
    print(f"  tokens : {' | '.join(show_token(t) for t in tokens)}")
    print(f"  ids    : {ids}")
    print(f"  {len(text)} characters, {len(text.split())} words -> {len(ids)} tokens")
    roundtrip = tokenizer.decode(ids)
    if roundtrip != text:
        print(f"  decoded: {roundtrip!r}   <- NOT the original text")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer and show how it splits text.")
    parser.add_argument("--merges", type=int, default=500, help="maximum number of merges to learn")
    parser.add_argument("--min-count", type=int, default=2, help="stop when the best pair is rarer than this")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="training text file")
    parser.add_argument("--text", action="append", help="text to tokenize (can repeat)")
    parser.add_argument("--show-merges", type=int, default=10, help="how many merges to print")
    args = parser.parse_args(argv)
    if args.merges < 0:
        parser.error("--merges cannot be negative")
    if not args.data.exists():
        parser.error(f"training file not found: {args.data}")
    text = args.data.read_text(encoding="utf-8")
    if not text.strip():
        parser.error(f"training file is empty: {args.data}")

    print(f"Training on {args.data.name}: {len(text):,} characters, {len(text.split()):,} words\n")
    tokenizer = train(text, args.merges, args.min_count, show=args.show_merges)
    learned = len(tokenizer.merges)
    print(f"\nLearned {learned} merges" + (f" (stopped early: no pair left that appears at least "
                                          f"{args.min_count} time(s))" if learned < args.merges else ""))
    print(f"Vocabulary size: {len(tokenizer.vocab)} tokens "
          f"(1 <unk> + {len(tokenizer.vocab) - 1 - learned} characters + {learned} merged pieces)")
    whole = tokenizer.encode(text)
    print(f"The whole training file becomes {len(whole):,} tokens "
          f"({len(text) / len(whole):.1f} characters per token)\n")

    for sample in args.text or DEMO_TEXTS:
        describe(tokenizer, sample)

    tokenizer.save(DEFAULT_SAVE)
    print(f"Tokenizer saved to {DEFAULT_SAVE.relative_to(HERE.parent)}")
    return tokenizer


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")  # don't crash if the terminal can't show a character
    main()
