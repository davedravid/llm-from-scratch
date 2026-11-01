"""Tests for Phase 6a: instruction fine-tuning and DPO preference tuning.

Run from the project folder (about 20 seconds; trains small throwaway models):
    .venv\\Scripts\\python -m unittest discover -s phase6_finetuning -v
"""
import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path

import torch

import common
import finetune
import preference
import train as story_train
from model import Config, TinyGPT

TOK = common.bpe.train("About the king.\nThe king was happy.\nThe king was sad.\n" * 3, 20)


def quietly(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args)


class Encoding(unittest.TestCase):
    def test_prompt_tokens_are_not_graded(self):
        x, y = common.encode_pair(TOK, "About the king.", "The king was happy.", 32)
        prompt_len = len(TOK.encode("About the king."))
        self.assertEqual(y[:prompt_len - 1], [common.IGNORE] * (prompt_len - 1))
        self.assertNotIn(common.IGNORE, y[prompt_len - 1:])
        self.assertEqual(len(x), len(y))

    def test_example_longer_than_the_context_is_rejected(self):
        with self.assertRaises(ValueError):
            common.encode_pair(TOK, "About the king.", "The king was happy. " * 10, 32)

    def test_unknown_characters_are_rejected(self):
        with self.assertRaises(ValueError):
            common.encode_pair(TOK, "About the king?", "The king was happy.", 32)

    def test_padding_is_ignored(self):
        a = common.encode_pair(TOK, "About the king.", "The king was happy.", 32)
        b = common.encode_pair(TOK, "About the king.", "The king was sad.", 32)
        xs, ys = common.pad_batch([a, b])
        self.assertEqual(xs.shape, ys.shape)
        self.assertTrue((ys[1, len(b[1]):] == common.IGNORE).all())

    def test_data_files_have_the_right_shape(self):
        self.assertEqual(len(common.read_blocks(common.DATA / "instructions.txt", 2)), 36)
        self.assertEqual(len(common.read_blocks(common.DATA / "preferences.txt", 3)), 26)


class ModeBugRegression(unittest.TestCase):
    """We found that evaluation helpers left models in training mode (dropout on), which made
    'deterministic' generation random. These tests stop that bug coming back."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = TinyGPT(Config(vocab_size=len(TOK.vocab), dropout=0.5)).eval()

    def test_response_loss_keeps_eval_mode(self):
        common.response_loss(self.model, [common.encode_pair(TOK, "About the king.", "The king was sad.", 32)])
        self.assertFalse(self.model.training)

    def test_story_evaluate_keeps_eval_mode(self):
        story_train.evaluate(self.model, torch.tensor(TOK.encode("The king was sad. " * 5)))
        self.assertFalse(self.model.training)


class DPO(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.policy = TinyGPT(Config(vocab_size=len(TOK.vocab)))
        self.reference = TinyGPT(Config(vocab_size=len(TOK.vocab)))
        self.reference.load_state_dict(self.policy.state_dict())
        self.chosen = [common.encode_pair(TOK, "About the king.", "The king was happy.", 32)]
        self.rejected = [common.encode_pair(TOK, "About the king.", "The king was sad.", 32)]

    def test_loss_is_log2_when_policy_equals_reference(self):
        self.policy.eval()
        loss = preference.dpo_loss(self.policy, self.reference.eval(), self.chosen, self.rejected, 0.5)
        self.assertAlmostEqual(loss.item(), math.log(2), places=4)

    def test_training_makes_the_chosen_answer_more_likely(self):
        self.policy.eval()
        self.reference.eval()
        before, _ = preference.preference_accuracy(self.policy, self.chosen, self.rejected)
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-2)
        for _ in range(20):
            loss = preference.dpo_loss(self.policy, self.reference, self.chosen, self.rejected, 0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        after, gap = preference.preference_accuracy(self.policy, self.chosen, self.rejected)
        self.assertEqual(after, 1.0)
        self.assertGreater(gap, 0)

    def test_logprob_matches_a_manual_calculation(self):
        self.policy.eval()
        x, y = self.chosen[0]
        with torch.no_grad():
            logp = torch.log_softmax(self.policy(torch.tensor([x]))[0][0], dim=-1)
            manual = sum(logp[i, t].item() for i, t in enumerate(y) if t != common.IGNORE)
            self.assertAlmostEqual(common.response_logprob(self.policy, self.chosen).item(), manual, places=4)


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.pretrained, cls.tuned, cls.pref = d / "pre.pt", d / "ft.pt", d / "pref.pt"
        quietly(story_train.main, ["--steps", "150", "--out", str(cls.pretrained)])
        cls.ft = quietly(finetune.main, ["--steps", "75", "--pretrained", str(cls.pretrained), "--out", str(cls.tuned)])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_finetuning_beats_the_base_model_on_held_out_answers(self):
        base, tok, _ = common.load_checkpoint(self.pretrained)
        pairs = common.read_blocks(common.DATA / "instructions.txt", 2)
        _, held = common.split_every(pairs, 6)
        held_ex = [common.encode_pair(tok, p, r, base.cfg.context_length) for p, r in held]
        self.assertLess(self.ft["held_out_loss"], common.response_loss(base, held_ex))

    def test_finetuned_model_keeps_the_same_shape_and_tokenizer(self):
        base, tok_a, _ = common.load_checkpoint(self.pretrained)
        tuned, tok_b, _ = common.load_checkpoint(self.tuned)
        self.assertEqual(base.cfg, tuned.cfg)
        self.assertEqual(tok_a.vocab, tok_b.vocab)
        self.assertEqual(base.count_parameters(), tuned.count_parameters())

    def test_preference_tuning_raises_held_out_accuracy(self):
        result = quietly(preference.main, ["--steps", "20", "--start", str(self.tuned), "--out", str(self.pref)])
        before, after = result["accuracy"]
        self.assertGreater(after, before)

    def test_missing_starting_model_gives_clear_error(self):
        with self.assertRaises(SystemExit):
            quietly(preference.main, ["--start", "nowhere.pt"])


if __name__ == "__main__":
    unittest.main()
