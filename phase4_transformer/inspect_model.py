"""Phase 4a: follow one sentence through an UNTRAINED transformer, step by step.

Shows the shapes at every stage, where the parameters are, the attention
weights, what the untrained model predicts, and a check of the causal mask.

Run (from the project folder):
    .venv\\Scripts\\python phase4_transformer\\inspect_model.py
Options: --text "..."  --layers 2  --dim 32  --heads 4  --no-causal  --seed 0
"""
import argparse
import math
from collections import defaultdict

import torch

from model import Config, TinyGPT
from tokens import bpe, load_tokenizer


def shape(t):
    return "(" + ", ".join(str(n) for n in t.shape) + ")"


def show_token(token):
    return bpe.show_token(token)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect an untrained tiny transformer.")
    parser.add_argument("--text", default="Once upon a time there was a king")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--no-causal", action="store_true", help="switch the causal mask off")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.dim % args.heads:
        parser.error("--dim must be divisible by --heads")

    tok = load_tokenizer()
    cfg = Config(vocab_size=len(tok.vocab), dim=args.dim, heads=args.heads,
                 layers=args.layers, causal=not args.no_causal)
    torch.manual_seed(args.seed)
    model = TinyGPT(cfg).eval()

    tokens = tok.tokenize(args.text)
    ids = torch.tensor([tok.encode(args.text)])
    if ids.shape[1] < 2:
        parser.error("--text needs at least two tokens")
    if ids.shape[1] > cfg.context_length:
        parser.error(f"--text is {ids.shape[1]} tokens; the context window is {cfg.context_length}")

    print("1. TEXT -> TOKENS -> IDS")
    print(f"   {' | '.join(show_token(t) for t in tokens)}")
    print(f"   ids {ids[0].tolist()}  shape {shape(ids)} = (batch of 1 sentence, {ids.shape[1]} tokens)\n")

    print("2. THE JOURNEY THROUGH THE MODEL (shapes)")
    with torch.no_grad():
        positions = torch.arange(ids.shape[1])
        tok_vecs = model.token_embedding(ids)
        pos_vecs = model.position_embedding(positions)
        x = tok_vecs + pos_vecs
        print(f"   token embeddings    {shape(tok_vecs)}  one {cfg.dim}-number vector per token")
        print(f"   + position vectors  {shape(pos_vecs)}  one vector per position 0..{ids.shape[1] - 1}")
        print(f"   = block input       {shape(x)}")
        for i, block in enumerate(model.blocks, 1):
            x = block(x)
            print(f"   after block {i}       {shape(x)}  same shape: blocks refine vectors, they don't resize them")
        logits = model.to_logits(model.final_norm(x))
        print(f"   logits              {shape(logits)}  a score for every one of the {cfg.vocab_size} "
              f"vocabulary tokens, at every position\n")

    print("3. WHERE THE PARAMETERS ARE")
    groups = defaultdict(int)
    for name, p in model.named_parameters():
        parts = name.split(".")
        key = ".".join(parts[:3]) if parts[0] == "blocks" else parts[0]
        groups[key.replace("norm1", "layer norms").replace("norm2", "layer norms")] += p.numel()
    total = model.count_parameters()
    for key, n in groups.items():
        print(f"   {key:<28} {n:>7,}  ({n / total:5.1%})")
    print(f"   {'TOTAL':<28} {total:>7,}\n")

    print("4. ATTENTION WEIGHTS, block 1, head 1 (row = token looking, column = token looked at)")
    weights = model.blocks[0].attention.last_weights[0, 0]
    labels = [show_token(t)[:7] for t in tokens]
    print("          " + "".join(f"{label:>8}" for label in labels))
    for label, row in zip(labels, weights):
        print(f"   {label:>7}" + "".join(f"{w:>8.2f}" for w in row.tolist()))
    print("   Untrained, so each token spreads its attention roughly evenly over what it may see.\n")

    print("5. WHAT THE UNTRAINED MODEL PREDICTS NEXT")
    probs = torch.softmax(logits[0, -1], dim=-1)
    top = torch.topk(probs, 5)
    print(f"   after {args.text!r}:")
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"     {show_token(tok.vocab[i]):<12} {p:.2%}")
    _, loss = model(ids[:, :-1], ids[:, 1:])
    print(f"   Loss on this sentence: {loss.item():.2f}. Pure guessing among {cfg.vocab_size} tokens "
          f"gives ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.2f}.\n")

    print("6. CAUSAL MASK CHECK: change the LAST token and see which positions' outputs change")
    changed = ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, _ = model(ids)
        b, _ = model(changed)
    diffs = (a - b).abs().amax(dim=-1)[0].tolist()
    for label, d in zip(labels, diffs):
        print(f"   {label:>7}: {'changed' if d > 1e-6 else 'unchanged'}  (max difference {d:.6f})")
    mask_state = "ON" if cfg.causal else "OFF"
    print(f"   Causal mask is {mask_state}: "
          + ("only the last position may change." if cfg.causal else "earlier positions can SEE the future."))


if __name__ == "__main__":
    main()
