"""Tests for Phase 4b: data split, batches, training, checkpoints and generation.

Run from the project folder (takes about half a minute):
    .venv\\Scripts\\python -m unittest discover -s phase4_transformer -v
"""
import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path

import torch

import data
import generate
import train
from model import Config, TinyGPT


def run_quietly(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args)


class Data(unittest.TestCase):
    def test_thirty_stories_are_loaded(self):
        self.assertEqual(len(data.load_stories()), 30)

    def test_split_has_no_overlap_and_keeps_everything(self):
        stories = data.load_stories()
        tr, val = data.split_stories(stories, 3)
        self.assertEqual(len(val), 3)
        self.assertFalse(set(tr) & set(val))
        self.assertEqual(sorted(tr + val), sorted(stories))

    def test_invalid_validation_count_rejected(self):
        with self.assertRaises(ValueError):
            data.split_stories(["a", "b"], 2)

    def test_targets_are_inputs_shifted_by_one(self):
        ids = torch.arange(100)
        x, y = data.get_batch(ids, 4, 10, torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))
        self.assertTrue(torch.equal(y[:, -1], x[:, -1] + 1))

    def test_too_little_data_rejected(self):
        with self.assertRaises(ValueError):
            data.get_batch(torch.arange(10), 4, 10, torch.Generator())


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "model.pt"
        cls.history, cls.best = run_quietly(train.main, ["--steps", "300", "--out", str(cls.path)])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_validation_loss_beats_random_guessing(self):
        _, tokenizer, _ = generate.load_checkpoint(self.path)
        self.assertLess(self.best["val_loss"], math.log(len(tokenizer.vocab)) - 1)

    def test_saved_model_is_the_best_validation_step(self):
        _, _, info = generate.load_checkpoint(self.path)
        self.assertEqual(info["step"], self.best["step"])
        self.assertEqual(min(h["val_loss"] for h in info["history"]), round(self.best["val_loss"], 4))

    def test_checkpoint_reproduces_the_same_predictions(self):
        model, tokenizer, _ = generate.load_checkpoint(self.path)
        again, _, _ = generate.load_checkpoint(self.path)
        ids = torch.tensor([tokenizer.encode("Once upon a time")])
        torch.testing.assert_close(model(ids)[0], again(ids)[0])

    def test_greedy_generation_is_repeatable(self):
        model, tokenizer, _ = generate.load_checkpoint(self.path)
        a = generate.continue_text(model, tokenizer, "The king", 20)
        b = generate.continue_text(model, tokenizer, "The king", 20)
        self.assertEqual(a, b)

    def test_generation_can_run_past_the_context_window(self):
        model, tokenizer, _ = generate.load_checkpoint(self.path)
        ids = torch.tensor([tokenizer.encode("Once upon a time")])
        out = model.generate(ids, model.cfg.context_length + 10)
        self.assertEqual(out.shape[1], ids.shape[1] + model.cfg.context_length + 10)

    def test_empty_prompt_rejected(self):
        model, tokenizer, _ = generate.load_checkpoint(self.path)
        with self.assertRaises(ValueError):
            generate.continue_text(model, tokenizer, "", 5)


class Overfitting(unittest.TestCase):
    def test_long_training_without_dropout_memorises(self):
        # Big vocabulary, no dropout, many steps: training loss collapses, validation loss explodes.
        with tempfile.TemporaryDirectory() as tmp:
            history, _ = run_quietly(train.main, ["--steps", "400", "--merges", "500", "--dropout", "0",
                                                  "--keep", "last", "--out", str(Path(tmp) / "m.pt")])
        last = history[-1]
        self.assertLess(last["train_loss"], 1.0)
        self.assertGreater(last["val_loss"], last["train_loss"] + 3)


class Helpers(unittest.TestCase):
    def test_copied_run_finds_the_longest_shared_words(self):
        run = generate.longest_copied_run("and then the king smiled at me", "Then the king smiled. The end.")
        self.assertEqual(run, ["then", "the", "king", "smiled"])

    def test_copied_run_is_empty_for_new_words(self):
        self.assertEqual(generate.longest_copied_run("zebra quantum", "the king"), [])

    def test_dropout_only_acts_during_training(self):
        torch.manual_seed(0)
        model = TinyGPT(Config(vocab_size=20, dropout=0.5))
        ids = torch.tensor([[1, 2, 3, 4]])
        model.eval()
        torch.testing.assert_close(model(ids)[0], model(ids)[0])
        model.train()
        self.assertFalse(torch.allclose(model(ids)[0], model(ids)[0]))

    def test_missing_checkpoint_gives_clear_error(self):
        with self.assertRaises(SystemExit):
            run_quietly(generate.main, ["--checkpoint", "nowhere.pt"])


if __name__ == "__main__":
    unittest.main()
