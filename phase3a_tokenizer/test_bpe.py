"""Tests for the Phase 3a BPE tokenizer.

Run from the project folder:
    .venv\\Scripts\\python -m unittest discover -s phase3a_tokenizer -v
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import bpe

STORIES = bpe.DEFAULT_DATA.read_text(encoding="utf-8")


class Training(unittest.TestCase):
    def test_tiny_example_from_the_video(self):
        # "the" appears twice, so its pieces get merged first.
        tok = bpe.train("the cat sat on the mat", num_merges=3, min_count=2)
        self.assertEqual(tok.tokenize("the"), ["the"])
        self.assertEqual(tok.tokenize(" mat"), [" ", "m", "at"])  # "at" is in cat, sat, mat

    def test_zero_merges_means_single_characters(self):
        tok = bpe.train(STORIES, num_merges=0)
        self.assertEqual(tok.tokenize("king"), ["k", "i", "n", "g"])

    def test_more_merges_means_fewer_tokens(self):
        counts = [len(bpe.train(STORIES, m).encode(STORIES)) for m in (0, 50, 200)]
        self.assertGreater(counts[0], counts[1])
        self.assertGreater(counts[1], counts[2])

    def test_training_is_repeatable(self):
        self.assertEqual(bpe.train(STORIES, 100).merges, bpe.train(STORIES, 100).merges)

    def test_tokens_never_span_two_words(self):
        tok = bpe.train(STORIES, 500)
        for token in tok.vocab:
            self.assertNotIn(" ", token.lstrip(" "), f"token {token!r} spans a word boundary")


class EncodeDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = bpe.train(STORIES, 500)

    def test_roundtrip_on_training_text(self):
        self.assertEqual(self.tok.decode(self.tok.encode(STORIES)), STORIES)

    def test_roundtrip_on_new_text_with_known_characters(self):
        text = "The astronaut zoomed past the moon."
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_common_words_are_single_tokens(self):
        self.assertEqual(self.tok.tokenize("Once upon a time"), ["Once", " upon", " a", " time"])

    def test_empty_text_gives_no_tokens(self):
        self.assertEqual(self.tok.encode(""), [])

    def test_save_and_load_give_the_same_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tok.json"
            self.tok.save(path)
            loaded = bpe.Tokenizer.load(path)
        self.assertEqual(loaded.encode(STORIES), self.tok.encode(STORIES))


class KnownLimitations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = bpe.train(STORIES, 500)

    def test_unseen_character_becomes_unk_and_is_lost(self):
        # The stories never contain a capital J or an emoji, so both are lost on decode.
        # Real tokenizers (GPT-2 onwards) work on bytes instead, so nothing is ever unknown.
        self.assertEqual(self.tok.decode(self.tok.encode("Jam 🙂")), "<unk>am <unk>")

    def test_unfamiliar_words_cost_more_tokens(self):
        familiar = self.tok.encode("the king and the queen")
        unfamiliar = self.tok.encode("the astronaut and the robot")
        self.assertGreater(len(unfamiliar), len(familiar))

    def test_other_languages_cost_more_tokens_per_word(self):
        english = self.tok.encode("Once upon a time there was a king.")
        french = self.tok.encode("Il etait une fois un roi.")
        self.assertGreater(len(french) / 6, len(english) / 8)


class CommandLine(unittest.TestCase):
    def run_main(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return bpe.main(list(args))

    def test_negative_merges_rejected(self):
        with self.assertRaises(SystemExit):
            self.run_main("--merges", "-1")

    def test_missing_training_file_rejected(self):
        with self.assertRaises(SystemExit):
            self.run_main("--data", "does_not_exist.txt")


if __name__ == "__main__":
    unittest.main()
