"""Phase 2: a neural network that learns XOR, trained with backpropagation.

We write the network and the training ourselves. numpy only does the array
arithmetic (multiplying and adding many numbers at once).

Run (from the project folder):
    .venv\\Scripts\\python phase2_neural_net\\xor.py --layers single
    .venv\\Scripts\\python phase2_neural_net\\xor.py --layers multi
Options: --task xor|and|or  --hidden 4  --epochs 5000  --lr 1.0  --seed 0
"""
import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Part 1: the training data. 4 examples, each with 2 inputs and 1 target.
# ---------------------------------------------------------------------------

INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

TARGETS = {
    "xor": [0, 1, 1, 0],  # 1 if exactly one input is 1
    "and": [0, 0, 0, 1],  # 1 if both inputs are 1
    "or":  [0, 1, 1, 1],  # 1 if at least one input is 1
}

WEIGHTS_DIR = Path(__file__).parent / "weights"


def sigmoid(z):
    """Squash any number into the range 0..1. Large positive -> ~1, large negative -> ~0."""
    return 1 / (1 + np.exp(-z))


def mse(outputs, targets):
    """Loss: the average of (output - target) squared. 0 means perfect."""
    return float(np.mean((outputs - targets) ** 2))


# ---------------------------------------------------------------------------
# Part 2: the two networks. Each has a "forward" step (inputs -> output)
# and a "train" loop that adjusts the weights.
#
# `@` is matrix multiplication: for each example, multiply every input by
# its weight and add them up, for all neurons at once.
# ---------------------------------------------------------------------------

def init_single(rng):
    # Random starting weights: 2 inputs -> 1 output, plus 1 bias.
    return {"w": rng.normal(size=(2, 1)), "b": np.zeros(1)}


def forward_single(params, x):
    return sigmoid(x @ params["w"] + params["b"])


def train_single(params, targets, epochs, lr, log_every):
    history = []
    for epoch in range(epochs + 1):
        out = forward_single(params, INPUTS)                 # 1. predict
        history.append(mse(out, targets))                    # 2. measure how wrong
        if epoch % log_every == 0:
            print(f"  epoch {epoch:>6}  loss {history[-1]:.4f}")
        # 3. Backpropagation: which way should each weight move to reduce the loss?
        # (out - targets) is the error; out*(1-out) is how much the sigmoid
        # changes the signal at this point (its derivative).
        delta = (out - targets) * out * (1 - out)
        # 4. Nudge each weight a small step (lr) in the direction that lowers the loss.
        params["w"] -= lr * INPUTS.T @ delta
        params["b"] -= lr * delta.sum(axis=0)
    return params, history


def init_multi(hidden_size, rng):
    # Random starting weights: 2 inputs -> `hidden_size` hidden neurons -> 1 output.
    return {
        "w1": rng.normal(size=(2, hidden_size)), "b1": np.zeros(hidden_size),
        "w2": rng.normal(size=(hidden_size, 1)), "b2": np.zeros(1),
    }


def forward_multi(params, x):
    hidden = sigmoid(x @ params["w1"] + params["b1"])
    out = sigmoid(hidden @ params["w2"] + params["b2"])
    return hidden, out


def train_multi(params, targets, epochs, lr, log_every):
    history = []
    for epoch in range(epochs + 1):
        hidden, out = forward_multi(params, INPUTS)          # 1. predict
        history.append(mse(out, targets))                    # 2. measure how wrong
        if epoch % log_every == 0:
            print(f"  epoch {epoch:>6}  loss {history[-1]:.4f}")
        # 3. Backpropagation, starting at the output and working backwards:
        delta_out = (out - targets) * out * (1 - out)
        # Pass the error back through the output weights to blame each hidden neuron.
        delta_hidden = (delta_out @ params["w2"].T) * hidden * (1 - hidden)
        # 4. Nudge every weight.
        params["w2"] -= lr * hidden.T @ delta_out
        params["b2"] -= lr * delta_out.sum(axis=0)
        params["w1"] -= lr * INPUTS.T @ delta_hidden
        params["b1"] -= lr * delta_hidden.sum(axis=0)
    return params, history


# ---------------------------------------------------------------------------
# Part 3: showing results and saving the model file.
# ---------------------------------------------------------------------------

def predict(layers, params):
    if layers == "single":
        return forward_single(params, INPUTS)
    return forward_multi(params, INPUTS)[1]


def show_predictions(outputs, targets):
    print("  input  target  output  rounded")
    correct = 0
    for x, t, o in zip(INPUTS, targets, outputs):
        rounded = int(o[0] >= 0.5)
        ok = rounded == int(t[0])
        correct += ok
        print(f"  {int(x[0])} {int(x[1])}    {int(t[0])}       {o[0]:.3f}   {rounded}   {'ok' if ok else 'WRONG'}")
    print(f"  Correct: {correct}/4")
    return correct


def count_params(params):
    return sum(p.size for p in params.values())


def save_weights(params, path, info):
    path.parent.mkdir(exist_ok=True)
    # "+ 0.0" turns -0.0 (a tiny negative number rounded away) into plain 0.0.
    data = dict(info, weights={name: (np.round(p, 4) + 0.0).tolist() for name, p in params.items()})
    path.write_text(json.dumps(data, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a tiny neural network on XOR, AND or OR.")
    parser.add_argument("--layers", choices=["single", "multi"], default="multi")
    parser.add_argument("--task", choices=list(TARGETS), default="xor")
    parser.add_argument("--hidden", type=int, default=4, help="hidden neurons (multi only)")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1.0, help="learning rate: size of each weight nudge")
    parser.add_argument("--seed", type=int, default=0, help="changes the random starting weights")
    args = parser.parse_args(argv)
    if args.hidden < 1:
        parser.error("--hidden must be at least 1")
    if args.epochs < 0:
        parser.error("--epochs cannot be negative")

    targets = np.array(TARGETS[args.task], dtype=float).reshape(-1, 1)
    rng = np.random.default_rng(args.seed)
    log_every = max(1, args.epochs // 10)

    shape = "2 -> 1" if args.layers == "single" else f"2 -> {args.hidden} -> 1"
    print(f"Training a {args.layers}-layer network ({shape}) on {args.task.upper()}")

    params = init_single(rng) if args.layers == "single" else init_multi(args.hidden, rng)
    print("\nBefore training (random weights):")
    show_predictions(predict(args.layers, params), targets)

    print("\nTraining:")
    train = train_single if args.layers == "single" else train_multi
    params, history = train(params, targets, args.epochs, args.lr, log_every)

    print("\nAfter training:")
    correct = show_predictions(predict(args.layers, params), targets)

    path = WEIGHTS_DIR / f"{args.layers}_{args.task}.json"
    save_weights(params, path, {"layers": args.layers, "task": args.task, "epochs": args.epochs,
                                "final_loss": round(history[-1], 6), "parameters": count_params(params)})
    print(f"\nParameters (numbers the network learned): {count_params(params)}")
    print(f"Weights saved to {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
    return correct, history


if __name__ == "__main__":
    main()
