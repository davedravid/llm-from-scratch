"""Tests for the Phase 3b word2vec embeddings.

Run from the project folder:
    .venv\\Scripts\\python -m unittest discover -s phase3b_embeddings -v
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

import word2vec as w2v


def train_embeddings(epochs=300, dim=16, seed=0):
    sentences = w2v.read_sentences(w2v.DEFAULT_DATA)
    words, counts = w2v.build_vocab(sentences)
    word_id = {w: i for i, w in enumerate(words)}
    pairs = w2v.make_pairs(sentences, word_id, window=2)
    counts_by_id = np.array([counts[w] for w in words], dtype=float)
    vectors, history = w2v.train(pairs, len(words), counts_by_id, dim, epochs, 5, 0.05,
                                 np.random.default_rng(seed))
    return w2v.Embeddings(words, vectors), history


class Pairs(unittest.TestCase):
    def test_skipgram_pairs_for_a_short_sentence(self):
        word_id = {"the": 0, "king": 1, "rules": 2}
        pairs = w2v.make_pairs([["the", "king", "rules"]], word_id, window=1)
        self.assertEqual(sorted(map(tuple, pairs.tolist())), [(0, 1), (1, 0), (1, 2), (2, 1)])

    def test_pairs_never_cross_sentences(self):
        word_id = {"a": 0, "b": 1}
        pairs = w2v.make_pairs([["a"], ["b"]], word_id, window=5)
        self.assertEqual(len(pairs), 0)


class TrainedEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.emb, cls.history = train_embeddings()

    def test_loss_goes_down(self):
        self.assertLess(self.history[-1], self.history[0])

    def test_related_words_are_more_similar_than_unrelated(self):
        self.assertGreater(self.emb.similarity("happy", "joyful"),
                           self.emb.similarity("happy", "refrigerator"))
        self.assertGreater(self.emb.similarity("king", "queen"),
                           self.emb.similarity("king", "banana"))

    def test_analogies(self):
        for a, b, c, expected in w2v.ANALOGIES:
            self.assertEqual(self.emb.analogy(a, b, c)[0][0], expected, f"{a} - {b} + {c}")

    def test_cosine_of_a_word_with_itself_is_one(self):
        self.assertAlmostEqual(self.emb.similarity("king", "king"), 1.0)

    def test_unknown_word_raises_a_clear_error(self):
        with self.assertRaises(KeyError):
            self.emb.vector("astronaut")

    def test_saved_file_reproduces_the_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "emb.json"
            self.emb.save(path, {})
            loaded = w2v.Embeddings.load(path)
        self.assertAlmostEqual(loaded.similarity("king", "queen"), self.emb.similarity("king", "queen"), places=3)


class KnownLimitations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.emb, _ = train_embeddings()

    def test_opposites_end_up_close(self):
        # "happy" and "sad" appear in the same kinds of sentences, so they get similar vectors.
        # Embeddings capture how words are USED, which isn't the same as what they MEAN.
        self.assertGreater(self.emb.similarity("happy", "sad"), 0.5)

    def test_analogy_without_excluding_inputs_returns_an_input(self):
        target = self.emb.vector("king") - self.emb.vector("man") + self.emb.vector("woman")
        self.assertEqual(self.emb.nearest(target, k=1)[0][0], "king")

    def test_too_few_epochs_fails_the_analogies(self):
        emb, _ = train_embeddings(epochs=5)
        solved = sum(emb.analogy(a, b, c)[0][0] == exp for a, b, c, exp in w2v.ANALOGIES)
        self.assertLess(solved, 3)

    def test_two_dimensions_is_too_small(self):
        emb, _ = train_embeddings(dim=2)
        solved = sum(emb.analogy(a, b, c)[0][0] == exp for a, b, c, exp in w2v.ANALOGIES)
        self.assertLess(solved, 3)


class CommandLine(unittest.TestCase):
    def run_main(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return w2v.main(list(args))

    def test_unknown_analogy_word_rejected_before_training(self):
        with self.assertRaises(SystemExit):
            self.run_main("--analogy", "king", "man", "astronaut")

    def test_zero_dimensions_rejected(self):
        with self.assertRaises(SystemExit):
            self.run_main("--dim", "0")


if __name__ == "__main__":
    unittest.main()
