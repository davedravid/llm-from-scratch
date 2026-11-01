"""Tests for the Phase 2 XOR network.

Run from the project folder:
    .venv\\Scripts\\python -m unittest discover -s phase2_neural_net -v
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import xor


def targets_for(task):
    return np.array(xor.TARGETS[task], dtype=float).reshape(-1, 1)


def quietly(fn, *args):
    """Run fn without printing the training log."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def accuracy(layers, params, task):
    rounded = (xor.predict(layers, params) >= 0.5).astype(int).ravel()
    return int((rounded == np.array(xor.TARGETS[task])).sum())


def train(layers, task="xor", hidden=4, epochs=5000, lr=1.0, seed=0):
    rng = np.random.default_rng(seed)
    params = xor.init_single(rng) if layers == "single" else xor.init_multi(hidden, rng)
    fn = xor.train_single if layers == "single" else xor.train_multi
    return quietly(fn, params, targets_for(task), epochs, lr, 10**9)


class Learning(unittest.TestCase):
    def test_multi_layer_learns_xor(self):
        params, history = train("multi")
        self.assertEqual(accuracy("multi", params, "xor"), 4)
        self.assertLess(history[-1], 0.01)

    def test_loss_goes_down_during_training(self):
        _, history = train("multi")
        self.assertLess(history[-1], history[0])

    def test_single_layer_learns_and_and_or(self):
        # The single-layer network isn't broken: it handles problems a straight line can split.
        for task in ("and", "or"):
            params, _ = train("single", task)
            self.assertEqual(accuracy("single", params, task), 4, task)


class ExpectedFailures(unittest.TestCase):
    def test_single_layer_cannot_learn_xor(self):
        params, history = train("single", "xor")
        self.assertLess(accuracy("single", params, "xor"), 4)
        self.assertAlmostEqual(history[-1], 0.25, places=2)  # stuck at "answer 0.5 for everything"

    def test_one_hidden_neuron_is_not_enough(self):
        params, _ = train("multi", hidden=1)
        self.assertLess(accuracy("multi", params, "xor"), 4)

    def test_learning_rate_zero_learns_nothing(self):
        _, history = train("multi", lr=0)
        self.assertEqual(history[0], history[-1])

    def test_learning_rate_too_big_fails(self):
        with np.errstate(over="ignore"):  # huge steps overflow; that's part of the failure
            params, _ = train("multi", lr=100)
        self.assertLess(accuracy("multi", params, "xor"), 4)

    def test_invalid_hidden_size_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            xor.main(["--hidden", "0"])


class ModelFile(unittest.TestCase):
    def test_parameter_counts_match_the_video(self):
        self.assertEqual(xor.count_params(train("single")[0]), 3)   # 2 weights + 1 bias
        self.assertEqual(xor.count_params(train("multi")[0]), 17)   # 8 + 4 + 4 + 1

    def test_saved_file_alone_reproduces_the_predictions(self):
        # A "model" is just this file of numbers: load it and you get the same answers.
        params, _ = train("multi")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            xor.save_weights(params, path, {"layers": "multi"})
            loaded = {k: np.array(v) for k, v in json.loads(path.read_text())["weights"].items()}
        np.testing.assert_allclose(xor.predict("multi", loaded), xor.predict("multi", params), atol=1e-3)


if __name__ == "__main__":
    unittest.main()
