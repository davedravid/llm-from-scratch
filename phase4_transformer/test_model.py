"""Tests for the Phase 4a transformer architecture.

Run from the project folder:
    .venv\\Scripts\\python -m unittest discover -s phase4_transformer -v
"""
import math
import unittest

import numpy as np
import torch

import attention_by_hand
from model import Config, TinyGPT

VOCAB = 50


def make_model(**overrides):
    torch.manual_seed(0)
    return TinyGPT(Config(vocab_size=VOCAB, **overrides)).eval()


def random_ids(length, seed=0):
    return torch.randint(0, VOCAB, (1, length), generator=torch.Generator().manual_seed(seed))


class AttentionByHand(unittest.TestCase):
    def test_bank_leans_towards_its_context(self):
        river, _ = attention_by_hand.attention(["the", "river", "bank"], verbose=False)
        money, _ = attention_by_hand.attention(["the", "money", "bank"], verbose=False)
        self.assertGreater(river[-1][0], river[-1][1])  # more nature than money
        self.assertGreater(money[-1][1], money[-1][0])  # more money than nature

    def test_weights_sum_to_one_and_respect_the_mask(self):
        _, weights = attention_by_hand.attention(["the", "river", "bank"], verbose=False)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        self.assertTrue(np.all(np.triu(weights, k=1) == 0))


class Shapes(unittest.TestCase):
    def test_logits_have_one_score_per_vocab_token_per_position(self):
        logits, loss = make_model()(random_ids(10))
        self.assertEqual(tuple(logits.shape), (1, 10, VOCAB))
        self.assertIsNone(loss)

    def test_parameter_count_matches_the_formula(self):
        d, v, t, layers = 32, VOCAB, 32, 2
        attention = d * 3 * d + (d * d + d)
        feed_forward = (d * 4 * d + 4 * d) + (4 * d * d + d)
        norms = 2 * (2 * d)
        expected = v * d + t * d + layers * (attention + feed_forward + norms) + 2 * d + (d * v + v)
        self.assertEqual(make_model().count_parameters(), expected)


class CausalMask(unittest.TestCase):
    def outputs_that_change_when_last_token_changes(self, model):
        ids = random_ids(8)
        changed = ids.clone()
        changed[0, -1] = (changed[0, -1] + 1) % VOCAB
        with torch.no_grad():
            diff = (model(ids)[0] - model(changed)[0]).abs().amax(dim=-1)[0]
        return diff > 1e-6

    def test_with_mask_earlier_positions_cannot_see_the_future(self):
        changed = self.outputs_that_change_when_last_token_changes(make_model())
        self.assertEqual(changed.tolist(), [False] * 7 + [True])

    def test_without_mask_the_future_leaks_backwards(self):
        changed = self.outputs_that_change_when_last_token_changes(make_model(causal=False))
        self.assertTrue(changed[:-1].all())

    def test_attention_rows_sum_to_one_with_zeros_above_the_diagonal(self):
        model = make_model()
        model(random_ids(6))
        w = model.blocks[0].attention.last_weights[0]           # (heads, 6, 6)
        torch.testing.assert_close(w.sum(dim=-1), torch.ones(4, 6))
        self.assertTrue(torch.all(torch.triu(w, diagonal=1) == 0))


class Loss(unittest.TestCase):
    def test_untrained_loss_is_close_to_random_guessing(self):
        ids = random_ids(33)
        _, loss = make_model()(ids[:, :-1], ids[:, 1:])
        self.assertAlmostEqual(loss.item(), math.log(VOCAB), delta=0.7)

    def test_can_memorise_one_short_sequence(self):
        # A standard sanity check before real training: if a model can't learn ONE sequence by heart,
        # something is broken. (Memorising isn't the goal; it just proves learning works.)
        torch.manual_seed(0)
        model = TinyGPT(Config(vocab_size=VOCAB))
        ids = random_ids(17)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        for _ in range(150):
            _, loss = model(ids[:, :-1], ids[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), 0.1)


class BadInput(unittest.TestCase):
    def test_too_long_input_is_rejected(self):
        with self.assertRaises(ValueError):
            make_model()(random_ids(33))

    def test_token_id_outside_vocabulary_is_rejected(self):
        with self.assertRaises(ValueError):
            make_model()(torch.tensor([[1, 2, VOCAB]]))

    def test_dim_not_divisible_by_heads_is_rejected(self):
        with self.assertRaises(AssertionError):
            make_model(dim=30, heads=4)


if __name__ == "__main__":
    unittest.main()
