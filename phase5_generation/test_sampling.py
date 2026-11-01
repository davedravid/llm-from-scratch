"""Tests for Phase 5a: softmax, temperature, top-p, sampling, the context window and text metrics.

Run from the project folder:
    .venv\\Scripts\\python -m unittest discover -s phase5_generation -v
"""
import unittest

import torch

import metrics
import sampling
from model import Config, TinyGPT

LOGITS = torch.tensor([3.0, 2.0, 1.0, 0.0])


class Temperature(unittest.TestCase):
    def test_probabilities_add_up_to_one(self):
        for t in (0.0, 0.5, 1.0, 2.0):
            self.assertAlmostEqual(sampling.softmax_with_temperature(LOGITS, t).sum().item(), 1.0, places=5)

    def test_temperature_one_is_plain_softmax(self):
        torch.testing.assert_close(sampling.softmax_with_temperature(LOGITS, 1.0), torch.softmax(LOGITS, -1))

    def test_low_temperature_sharpens_high_temperature_flattens(self):
        top = [sampling.softmax_with_temperature(LOGITS, t)[0].item() for t in (0.5, 1.0, 2.0)]
        self.assertGreater(top[0], top[1])
        self.assertGreater(top[1], top[2])

    def test_temperature_zero_puts_everything_on_the_top_token(self):
        self.assertEqual(sampling.softmax_with_temperature(LOGITS, 0.0).tolist(), [1.0, 0.0, 0.0, 0.0])


class TopP(unittest.TestCase):
    def test_confident_model_keeps_one_token(self):
        probs = torch.tensor([0.95, 0.03, 0.02])
        self.assertEqual((sampling.top_p_filter(probs, 0.9) > 0).sum().item(), 1)

    def test_unsure_model_keeps_many_tokens(self):
        # 200 equally likely tokens: top_p=0.9 needs 180 of them, not "90% of the top".
        probs = torch.full((200,), 1 / 200)
        self.assertEqual((sampling.top_p_filter(probs, 0.9) > 0).sum().item(), 180)

    def test_kept_tokens_are_rescaled_to_add_up_to_one(self):
        filtered = sampling.top_p_filter(torch.softmax(LOGITS, -1), 0.8)
        self.assertAlmostEqual(filtered.sum().item(), 1.0, places=5)

    def test_top_token_is_always_kept(self):
        filtered = sampling.top_p_filter(torch.softmax(LOGITS, -1), 0.0001)
        self.assertEqual(filtered.tolist(), [1.0, 0.0, 0.0, 0.0])

    def test_top_p_one_keeps_everything(self):
        self.assertTrue((sampling.top_p_filter(torch.softmax(LOGITS, -1), 1.0) > 0).all())


class Sampling(unittest.TestCase):
    def test_same_seed_gives_same_tokens(self):
        a = [sampling.sample_next(LOGITS, 1.0, 1.0, torch.Generator().manual_seed(7)) for _ in range(5)]
        b = [sampling.sample_next(LOGITS, 1.0, 1.0, torch.Generator().manual_seed(7)) for _ in range(5)]
        self.assertEqual(a, b)

    def test_picks_follow_the_probabilities(self):
        gen = torch.Generator().manual_seed(0)
        picks = [sampling.sample_next(torch.log(torch.tensor([0.7, 0.3])), 1.0, 1.0, gen) for _ in range(4000)]
        self.assertAlmostEqual(picks.count(0) / len(picks), 0.7, delta=0.03)

    def test_filtered_out_tokens_are_never_picked(self):
        gen = torch.Generator().manual_seed(0)
        picks = {sampling.sample_next(LOGITS, 1.0, 0.6, gen) for _ in range(500)}
        self.assertTrue(picks <= {0, 1})

    def test_invalid_settings_rejected(self):
        for t, p in [(-0.1, 1.0), (1.0, 0.0), (1.0, 1.5)]:
            with self.assertRaises(ValueError):
                sampling.sample_next(LOGITS, t, p)


class ContextWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.tok = sampling.bpe.train("the king and the queen sat in the castle with the cat. " * 5, 30)
        cls.model = TinyGPT(Config(vocab_size=len(cls.tok.vocab), context_length=8)).eval()

    def test_fit_keeps_only_the_last_tokens(self):
        kept, dropped = sampling.fit_to_context(list(range(12)), 8)
        self.assertEqual(kept, list(range(4, 12)))
        self.assertEqual(dropped, [0, 1, 2, 3])

    def test_short_input_drops_nothing(self):
        self.assertEqual(sampling.fit_to_context([1, 2], 8), ([1, 2], []))

    def test_prompt_longer_than_the_window_still_generates(self):
        text = sampling.generate(self.model, self.tok, "the king and the queen sat in the castle", 5, seed=0)
        self.assertIsInstance(text, str)

    def test_text_outside_the_window_has_no_effect(self):
        tail = " the queen sat in the castle"
        self.assertLessEqual(len(self.tok.encode(tail)), 8)
        a = sampling.generate(self.model, self.tok, "the cat and the king" * 3 + tail, 10, 1.0, 1.0, seed=0)
        b = sampling.generate(self.model, self.tok, "the castle the castle" * 3 + tail, 10, 1.0, 1.0, seed=0)
        self.assertEqual(a, b)

    def test_empty_prompt_rejected(self):
        with self.assertRaises(ValueError):
            sampling.generate(self.model, self.tok, "", 5)


class Metrics(unittest.TestCase):
    def test_repetition_rate(self):
        self.assertEqual(metrics.repetition_rate("a b c a b c"), 0.25)  # 1 of 4 three-word runs repeats
        self.assertEqual(metrics.repetition_rate("a b c d e"), 0.0)

    def test_real_word_rate(self):
        self.assertEqual(metrics.real_word_rate("the farmerily king", {"the", "king"}), 2 / 3)

    def test_empty_text_scores_zero(self):
        self.assertEqual(metrics.real_word_rate("", {"the"}), 0.0)
        self.assertEqual(metrics.repetition_rate(""), 0.0)

    def test_distinct_outputs(self):
        self.assertEqual(metrics.distinct_outputs(["a", "a", "b"]), 2)


if __name__ == "__main__":
    unittest.main()
