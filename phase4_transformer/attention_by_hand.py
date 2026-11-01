"""Phase 4a: one attention step, worked through by hand with tiny numbers.

The query/key/value vectors below are PICKED BY US to make the idea visible.
In a real model they are not picked: they come from learned weights (see model.py).

Run (from the project folder):
    .venv\\Scripts\\python phase4_transformer\\attention_by_hand.py
"""
import numpy as np

# Each token gets 3 small vectors (2 numbers each):
#   query = what this token is looking for
#   key   = what this token offers, to be matched against queries
#   value = the information it passes on if it's attended to
# Value numbers mean [nature-ness, money-ness].
TOKENS = {
    #          query        key          value
    "the":   ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
    "river": ([0.0, 0.0], [2.0, 0.0], [1.0, 0.0]),
    "money": ([0.0, 0.0], [0.0, 2.0], [0.0, 1.0]),
    "bank":  ([1.0, 1.0], [1.0, 1.0], [0.5, 0.5]),   # "bank" alone is ambiguous: half and half
}


def softmax(scores):
    """Turn any list of scores into positive weights that add up to 1."""
    e = np.exp(scores - np.max(scores))
    return e / e.sum()


def attention(sentence, verbose=True):
    q = np.array([TOKENS[w][0] for w in sentence])
    k = np.array([TOKENS[w][1] for w in sentence])
    v = np.array([TOKENS[w][2] for w in sentence])
    size = q.shape[1]
    outputs, all_weights = [], []
    for i, word in enumerate(sentence):
        # Causal mask: token i may only look at tokens 0..i (itself and earlier), never later ones.
        visible = range(i + 1)
        scores = np.array([q[i] @ k[j] for j in visible]) / np.sqrt(size)  # 1. match query vs keys
        weights = softmax(scores)                                          # 2. scores -> weights
        out = sum(weights[j] * v[j] for j in visible)                      # 3. weighted mix of values
        outputs.append(out)
        all_weights.append(list(weights) + [0.0] * (len(sentence) - i - 1))
    if verbose:
        print(f"Sentence: {' '.join(sentence)}")
        print("Attention weights (row = token doing the looking, column = token looked at):")
        print("          " + "".join(f"{w:>8}" for w in sentence))
        for word, row in zip(sentence, all_weights):
            print(f"  {word:>7} " + "".join(f"{x:>8.2f}" for x in row))
        word_out = outputs[-1]
        print(f"Output vector for '{sentence[-1]}': nature={word_out[0]:.2f}, money={word_out[1]:.2f}  "
              f"(it started as nature={v[-1][0]:.2f}, money={v[-1][1]:.2f})\n")
    return np.array(outputs), np.array(all_weights)


if __name__ == "__main__":
    print("Same word 'bank', two sentences. Watch its output vector change.\n")
    attention(["the", "river", "bank"])
    attention(["the", "money", "bank"])
    print("The zeros above the diagonal are the causal mask: no token can see a later one.")
