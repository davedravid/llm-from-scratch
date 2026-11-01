"""Phase 4b: train the tiny transformer on the stories.

Run (from the project folder):
    .venv\\Scripts\\python phase4_transformer\\train.py
Options: --steps 1000  --lr 0.003  --batch 32  --layers 2  --dim 32  --heads 4
         --merges 100  --dropout 0.2  --val-stories 3  --train-stories 27  --seed 0
"""
import argparse
import math
import time
from pathlib import Path

import torch

from data import load_stories, split_stories, get_batch, SEPARATOR
from generate import DEFAULT_CHECKPOINT, continue_text, load_checkpoint, longest_copied_run, save_checkpoint
from model import Config, TinyGPT
from tokens import bpe

PROMPTS = ["Once upon a time", "The king", "The dragon"]


@torch.no_grad()
def evaluate(model, ids):
    """Average loss over ALL of `ids`, in consecutive windows (no randomness, so runs compare fairly)."""
    was_training = model.training
    model.eval()
    length = model.cfg.context_length
    total, count = 0.0, 0
    for start in range(0, len(ids) - 1, length):
        chunk = ids[start:start + length + 1]
        if len(chunk) < 2:
            break
        _, loss = model(chunk[None, :-1], chunk[None, 1:])
        total += loss.item() * (len(chunk) - 1)
        count += len(chunk) - 1
    model.train(was_training)  # restore the previous mode (training mode would switch dropout on)
    return total / count


def show_samples(model, tokenizer, training_text, label):
    print(f"\n{label}")
    for prompt in PROMPTS:
        text = continue_text(model, tokenizer, prompt, 25)
        copied = len(longest_copied_run(text, training_text))
        print(f"  {prompt}{text!r}\n      longest word-for-word copy from training: {copied} words")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the tiny transformer on data/stories.txt.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
    parser.add_argument("--batch", type=int, default=32, help="windows of text per training step")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--merges", type=int, default=100, help="tokenizer merges (smaller = smaller vocabulary)")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val-stories", type=int, default=3, help="stories held back for validation")
    parser.add_argument("--train-stories", type=int, default=None, help="use only this many training stories")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--keep", choices=["best", "last"], default="best",
                        help="save the step with the best validation loss, or simply the last step")
    parser.add_argument("--out", type=Path, default=DEFAULT_CHECKPOINT, help="where to save the model")
    parser.add_argument("--extra-train", type=Path, help="a text file added to the training stories (Phase 5b)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.dim % args.heads:
        parser.error("--dim must be divisible by --heads")
    if args.steps < 1 or args.batch < 1:
        parser.error("--steps and --batch must be at least 1")

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    # 1. Data: split the stories, then build the tokenizer from the training stories only.
    try:
        train_stories, val_stories = split_stories(load_stories(), args.val_stories)
    except ValueError as error:
        parser.error(str(error))
    if args.train_stories is not None:
        if args.train_stories < 1:
            parser.error("--train-stories must be at least 1")
        train_stories = train_stories[:args.train_stories]
    if args.extra_train:
        if not args.extra_train.exists():
            parser.error(f"extra training file not found: {args.extra_train}")
        train_stories = train_stories + [args.extra_train.read_text(encoding="utf-8").strip()]
    train_text, val_text = SEPARATOR.join(train_stories), SEPARATOR.join(val_stories)
    tokenizer = bpe.train(train_text, num_merges=args.merges)
    train_ids = torch.tensor(tokenizer.encode(train_text))
    val_ids = torch.tensor(tokenizer.encode(val_text))
    unknown = (val_ids == tokenizer.id_of[bpe.UNK]).sum().item()

    cfg = Config(vocab_size=len(tokenizer.vocab), dim=args.dim, heads=args.heads, layers=args.layers,
                 dropout=args.dropout)
    if len(train_ids) <= cfg.context_length:
        parser.error("not enough training text for even one window; use more --train-stories")
    model = TinyGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Training stories: {len(train_stories)} ({len(train_ids):,} tokens)   "
          f"Validation stories: {len(val_stories)} ({len(val_ids):,} tokens, {unknown} unknown)")
    print(f"Vocabulary: {cfg.vocab_size} tokens   Model: {cfg.layers} blocks, {cfg.dim} dims, "
          f"{cfg.heads} heads = {model.count_parameters():,} parameters")
    print(f"Random guessing would score a loss of ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.2f}")
    print(f"Each step sees {args.batch} x {cfg.context_length} = {args.batch * cfg.context_length:,} tokens; "
          f"{args.steps} steps = about {args.steps * args.batch * cfg.context_length / len(train_ids):.0f} "
          f"passes over the training text")

    show_samples(model.eval(), tokenizer, train_text, "BEFORE TRAINING (random weights):")
    model.train()

    # 2. The training loop: same idea as Phase 2 (predict, measure, backpropagate, nudge),
    #    except PyTorch computes the gradients for us (loss.backward()) and AdamW does the nudging.
    # Loss is per TOKEN, so it can't be compared between tokenizers (predicting one letter is easier
    # than predicting a whole word). Loss per CHARACTER of text can be compared, so we show both.
    per_char = len(val_ids) / len(val_text)
    print(f"\n{'step':>6} {'train loss':>11} {'val loss':>9} {'val per char':>13}")
    best = {"val_loss": float("inf")}
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_ids, args.batch, cfg.context_length, generator)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            train_loss, val_loss = evaluate(model, train_ids), evaluate(model, val_ids)
            history.append({"step": step, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
                            "val_loss_per_char": round(val_loss * per_char, 4)})
            note = ""
            current = {"step": step, "val_loss": val_loss, "train_loss": train_loss,
                       "val_loss_per_char": val_loss * per_char}
            if val_loss < best["val_loss"]:
                best = current
                note = "  <- best validation so far" + (", saved" if args.keep == "best" else "")
            if (args.keep == "best" and best is current) or (args.keep == "last" and step == args.steps):
                save_checkpoint(args.out, model, tokenizer, dict(current, training_text=train_text))
            print(f"{step:>6} {train_loss:>11.3f} {val_loss:>9.3f} {val_loss * per_char:>13.3f}{note}")
    elapsed = time.time() - started

    final_train, final_val = history[-1]["train_loss"], history[-1]["val_loss"]
    print(f"\nTrained in {elapsed:.0f} seconds.")
    print(f"Last step:  train {final_train:.3f}, val {final_val:.3f}")
    print(f"Best step:  {best['step']} (train {best['train_loss']:.3f}, val {best['val_loss']:.3f}, "
          f"val per char {best['val_loss_per_char']:.3f})")
    print(f"Saved the {args.keep} step to {args.out}")

    # 3. Add the history to the saved file, then show what the saved model writes.
    model, tokenizer, info = load_checkpoint(args.out)
    info["history"] = history
    # Store paths as plain text: the safe loader (weights_only=True) refuses unfamiliar Python objects.
    info["settings"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    save_checkpoint(args.out, model, tokenizer, info)
    show_samples(model, tokenizer, train_text, f"AFTER TRAINING (saved model, step {info['step']}):")
    return history, best


if __name__ == "__main__":
    main()
