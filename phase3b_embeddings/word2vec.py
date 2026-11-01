"""Phase 3b: word embeddings with word2vec (skip-gram + negative sampling), from scratch.

Idea: "you shall know a word by the company it keeps". Words that appear in
similar sentences should end up with similar vectors. Training nudges the
vectors of neighbouring words together and of random word pairs apart.

Run (from the project folder):
    .venv\\Scripts\\python phase3b_embeddings\\word2vec.py
Options: --dim 16  --window 2  --epochs 300  --negatives 5  --lr 0.05  --seed 0
         --analogy king man woman   (computes king - man + woman)
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DEFAULT_DATA = HERE.parent / "data" / "relations.txt"
DEFAULT_SAVE = HERE / "embeddings.json"

# ---------------------------------------------------------------------------
# Part 1: text -> words -> training pairs.
# For embeddings we use simple whole-word tokens (lowercase, no punctuation) so the
# results are easy to read. In Phase 4 the transformer learns embeddings for BPE tokens.
# ---------------------------------------------------------------------------

def read_sentences(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [re.findall(r"[a-z]+", line.lower()) for line in lines if line.strip()]


def build_vocab(sentences):
    counts = Counter(word for sentence in sentences for word in sentence)
    words = sorted(counts)  # alphabetical, so IDs are the same every run
    return words, counts


def make_pairs(sentences, word_id, window):
    """Pair every word with each neighbour up to `window` words away (the skip-gram setup).

    "the king wears a crown", window 2, centre word "king" ->
        (king, the), (king, wears), (king, a)
    """
    pairs = []
    for sentence in sentences:
        ids = [word_id[w] for w in sentence]
        for i, centre in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    pairs.append((centre, ids[j]))
    return np.array(pairs)


# ---------------------------------------------------------------------------
# Part 2: training.
# Each word gets TWO vectors: an "embedding" (used when it's the centre word)
# and a "context vector" (used when it's the neighbour). The video only mentions
# one; the second is a training helper and is thrown away afterwards.
# ---------------------------------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train(pairs, vocab_size, counts_by_id, dim, epochs, negatives, lr, rng, batch_size=64, log_every=0):
    embed = rng.normal(scale=0.1, size=(vocab_size, dim))   # the vectors we keep
    context = np.zeros((vocab_size, dim))                   # the training helper
    # Negative samples are random words, picked more often if they're common
    # (count ** 0.75 is the word2vec paper's choice: common words, but not overwhelmingly).
    noise = counts_by_id ** 0.75
    noise /= noise.sum()
    history = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(pairs))
        total = 0.0
        for start in range(0, len(pairs), batch_size):
            batch = pairs[order[start:start + batch_size]]
            centres, contexts = batch[:, 0], batch[:, 1]
            negs = rng.choice(vocab_size, size=(len(batch), negatives), p=noise)

            c = embed[centres]                       # (batch, dim)
            o = context[contexts]                    # (batch, dim)
            n = context[negs]                        # (batch, negatives, dim)

            # Score = dot product: big when two vectors point the same way.
            # Real neighbours should score high (sigmoid -> 1), random words low (-> 0).
            pos = sigmoid(np.sum(c * o, axis=1))
            neg = sigmoid(np.einsum("bd,bkd->bk", c, n))
            total += float(-np.log(pos + 1e-9).sum() - np.log(1 - neg + 1e-9).sum())

            # Gradients: pull real pairs together, push random pairs apart.
            g_pos = (pos - 1)[:, None]               # negative: increase the score
            g_neg = neg[:, :, None]                  # positive: decrease the score
            grad_c = g_pos * o + np.sum(g_neg * n, axis=1)
            np.add.at(context, contexts, -lr * g_pos * c)
            np.add.at(context, negs, -lr * g_neg * c[:, None, :])
            np.add.at(embed, centres, -lr * grad_c)
        history.append(total / len(pairs))
        if log_every and (epoch == 1 or epoch % log_every == 0):
            print(f"  epoch {epoch:>4}  loss {history[-1]:.3f}")
    return embed, history


# ---------------------------------------------------------------------------
# Part 3: using the vectors.
# ---------------------------------------------------------------------------

class Embeddings:
    def __init__(self, words, vectors):
        self.words = list(words)
        self.vectors = np.asarray(vectors, dtype=float)
        self.index = {w: i for i, w in enumerate(self.words)}
        lengths = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.unit = self.vectors / np.maximum(lengths, 1e-9)   # same directions, length 1

    def vector(self, word):
        if word not in self.index:
            raise KeyError(f"'{word}' is not in the vocabulary (it never appeared in the training text)")
        return self.vectors[self.index[word]]

    def similarity(self, a, b):
        """Cosine similarity: 1 = same direction, 0 = perpendicular, -1 = opposite."""
        va, vb = self.vector(a), self.vector(b)
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))

    def nearest(self, vec, k=5, exclude=()):
        scores = self.unit @ (vec / np.linalg.norm(vec))
        ranked = [i for i in np.argsort(-scores) if self.words[i] not in exclude]
        return [(self.words[i], float(scores[i])) for i in ranked[:k]]

    def analogy(self, a, b, c, k=3):
        """a - b + c. E.g. king - man + woman. The three input words are excluded from the
        answers; otherwise the closest word is usually one of the inputs (standard practice,
        but worth knowing: it makes analogies look a bit better than they are)."""
        target = self.vector(a) - self.vector(b) + self.vector(c)
        return self.nearest(target, k, exclude={a, b, c})

    def save(self, path, info):
        data = dict(info, words=self.words, vectors=np.round(self.vectors, 4).tolist())
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text())
        return cls(data["words"], data["vectors"])


# ---------------------------------------------------------------------------
# Part 4: the demo.
# ---------------------------------------------------------------------------

SIMILARITY_PAIRS = [("king", "queen"), ("cat", "dog"), ("happy", "joyful"),
                    ("happy", "refrigerator"), ("king", "banana")]
NEIGHBOUR_WORDS = ["king", "kitten", "happy", "apple"]
ANALOGIES = [("king", "man", "woman", "queen"), ("queen", "woman", "man", "king"),
             ("prince", "boy", "girl", "princess"), ("kitten", "cat", "dog", "puppy"),
             ("man", "king", "queen", "woman")]


def show_similarities(emb):
    for a, b in SIMILARITY_PAIRS:
        print(f"  {a:>6} ~ {b:<13} {emb.similarity(a, b):+.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train word2vec embeddings and explore them.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dim", type=int, default=16, help="numbers per word vector")
    parser.add_argument("--window", type=int, default=2, help="how many neighbours on each side count")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--negatives", type=int, default=5, help="random words pushed away per pair")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analogy", nargs=3, metavar=("A", "B", "C"), help="compute A - B + C")
    args = parser.parse_args(argv)
    for name in ("dim", "window", "epochs", "negatives"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    if not args.data.exists():
        parser.error(f"training file not found: {args.data}")

    sentences = read_sentences(args.data)
    words, counts = build_vocab(sentences)
    if len(words) < 2:
        parser.error("the training file needs at least two different words")
    word_id = {w: i for i, w in enumerate(words)}
    if args.analogy:
        args.analogy = [w.lower() for w in args.analogy]
        unknown = [w for w in args.analogy if w not in word_id]
        if unknown:  # check before spending time on training
            parser.error(f"not in the vocabulary (never in the training text): {', '.join(unknown)}")
    pairs = make_pairs(sentences, word_id, args.window)
    print(f"Corpus: {len(sentences)} sentences, {sum(counts.values())} words, {len(words)} unique words")
    print(f"Training pairs (window {args.window}): {len(pairs)}")
    print(f"Parameters: {len(words)} words x {args.dim} numbers = {len(words) * args.dim} (plus the same again for the helper vectors)")

    rng = np.random.default_rng(args.seed)
    counts_by_id = np.array([counts[w] for w in words], dtype=float)
    demo_words = all(w in word_id for pair in SIMILARITY_PAIRS for w in pair)

    if demo_words and not args.analogy:
        before = Embeddings(words, np.random.default_rng(args.seed).normal(scale=0.1, size=(len(words), args.dim)))
        print("\nBefore training (random vectors), cosine similarity:")
        show_similarities(before)

    print("\nTraining:")
    vectors, _ = train(pairs, len(words), counts_by_id, args.dim, args.epochs, args.negatives,
                       args.lr, rng, log_every=max(1, args.epochs // 6))
    emb = Embeddings(words, vectors)

    if args.analogy:
        a, b, c = args.analogy
        results = emb.analogy(a, b, c)
        print(f"\n{a} - {b} + {c} = " + ", ".join(f"{w} ({s:.2f})" for w, s in results))
        return emb

    if demo_words:
        print("\nAfter training, cosine similarity:")
        show_similarities(emb)
        print("\nNearest neighbours:")
        for word in NEIGHBOUR_WORDS:
            near = emb.nearest(emb.vector(word), k=4, exclude={word})
            print(f"  {word:>7}: " + ", ".join(f"{w} ({s:.2f})" for w, s in near))
        print("\nAnalogies (a - b + c), top 3 answers:")
        for a, b, c, expected in ANALOGIES:
            results = emb.analogy(a, b, c)
            mark = "ok" if results[0][0] == expected else f"WRONG (expected {expected})"
            print(f"  {a} - {b} + {c} = " + ", ".join(f"{w} ({s:.2f})" for w, s in results) + f"   {mark}")

    emb.save(DEFAULT_SAVE, {"dim": args.dim, "window": args.window, "epochs": args.epochs, "seed": args.seed})
    print(f"\nEmbeddings saved to {DEFAULT_SAVE.relative_to(HERE.parent)}")
    return emb


if __name__ == "__main__":
    main()
