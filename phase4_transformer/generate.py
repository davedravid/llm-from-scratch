"""Phase 4b: load the trained model and generate text from a prompt.

Run (from the project folder), after train.py has saved a checkpoint:
    .venv\\Scripts\\python phase4_transformer\\generate.py --prompt "Once upon a time"
Options: --tokens 40  --attention  --checkpoint path
"""
import argparse
import re
from dataclasses import asdict
from pathlib import Path

import torch

from model import Config, TinyGPT
from tokens import bpe

HERE = Path(__file__).parent
DEFAULT_CHECKPOINT = HERE / "checkpoints" / "tinygpt.pt"


# ---------------------------------------------------------------------------
# Saving and loading. A checkpoint holds everything needed to use the model:
# its settings, its learned numbers, the tokenizer, and a record of training.
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, tokenizer, info):
    path.parent.mkdir(exist_ok=True)
    torch.save({
        "config": asdict(model.cfg),
        "weights": model.state_dict(),
        "tokenizer": {"vocab": tokenizer.vocab, "merges": [list(m) for m in tokenizer.merges]},
        "info": info,
    }, path)


def load_checkpoint(path):
    data = torch.load(path, weights_only=True)
    model = TinyGPT(Config(**data["config"]))
    model.load_state_dict(data["weights"])
    tokenizer = bpe.Tokenizer(data["tokenizer"]["vocab"], data["tokenizer"]["merges"])
    return model.eval(), tokenizer, data["info"]


# ---------------------------------------------------------------------------
# Generating, and checking whether the output is just copied from training.
# ---------------------------------------------------------------------------

def continue_text(model, tokenizer, prompt, new_tokens):
    ids = torch.tensor([tokenizer.encode(prompt)])
    if ids.shape[1] == 0:
        raise ValueError("the prompt is empty")
    out = model.generate(ids, new_tokens)
    return tokenizer.decode(out[0, ids.shape[1]:].tolist())


def words(text):
    return re.findall(r"[a-z']+", text.lower())


def longest_copied_run(generated, training_text):
    """The longest run of consecutive generated words that also appears, word for word,
    somewhere in the training text. A long run means the model is reciting, not composing."""
    gen, train = words(generated), " " + " ".join(words(training_text)) + " "
    best = []
    for start in range(len(gen)):
        for end in range(start + len(best) + 1, len(gen) + 1):
            if " " + " ".join(gen[start:end]) + " " in train:
                best = gen[start:end]
            else:
                break
    return best


def show_attention(model, tokenizer, prompt):
    ids = torch.tensor([tokenizer.encode(prompt)])
    model(ids)
    labels = [bpe.show_token(t)[:7] for t in tokenizer.tokenize(prompt)]
    for b, block in enumerate(model.blocks, 1):
        weights = block.attention.last_weights[0]
        for h in range(weights.shape[0]):
            print(f"\nBlock {b}, head {h + 1} (row = token looking, column = token looked at)")
            print("          " + "".join(f"{label:>8}" for label in labels))
            for label, row in zip(labels, weights[h]):
                print(f"   {label:>7}" + "".join(f"{w:>8.2f}" for w in row.tolist()))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate text with the trained tiny transformer.")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=40, help="how many new tokens to generate")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--attention", action="store_true", help="print every head's attention for the prompt")
    args = parser.parse_args(argv)
    if not args.checkpoint.exists():
        parser.error(f"no checkpoint at {args.checkpoint}. Run phase4_transformer\\train.py first.")
    if args.tokens < 1:
        parser.error("--tokens must be at least 1")

    model, tokenizer, info = load_checkpoint(args.checkpoint)
    prompt_len = len(tokenizer.encode(args.prompt))
    if prompt_len == 0:
        parser.error("--prompt is empty")
    if prompt_len > model.cfg.context_length:
        print(f"Note: the prompt is {prompt_len} tokens; the model only sees the last {model.cfg.context_length}.")

    print(f"Model from step {info['step']} (validation loss {info['val_loss']:.2f})\n")
    generated = continue_text(model, tokenizer, args.prompt, args.tokens)
    print(f"{args.prompt}{generated}\n")
    copied = longest_copied_run(generated, info["training_text"])
    print(f"Longest run copied word-for-word from the training stories: {len(copied)} words")
    if copied:
        print(f"  \"{' '.join(copied)}\"")
    if args.attention:
        show_attention(model, tokenizer, args.prompt)


if __name__ == "__main__":
    main()
