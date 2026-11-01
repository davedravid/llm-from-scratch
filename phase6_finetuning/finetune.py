"""Phase 6a: fine-tune the Phase 4b story model to answer instructions.

Same model, same tokenizer, same training loop. Only three things change:
  1. the data: instruction -> answer pairs instead of stories,
  2. the loss: only the answer tokens are graded,
  3. the starting point: the pretrained weights, not random ones (unless --from-scratch).

Run (from the project folder, about 20 seconds):
    .venv\\Scripts\\python phase6_finetuning\\finetune.py
Options: --steps 300  --lr 0.001  --from-scratch  --seed 0
"""
import argparse
import random
from pathlib import Path

import torch

from common import (CHECKPOINTS, DATA, PRETRAINED, encode_pair, first_line, generate, load_checkpoint,
                    pad_batch, read_blocks, response_loss, save_checkpoint, split_every)
from model import TinyGPT
import data as story_data
import train as story_train

OUT = CHECKPOINTS / "finetuned.pt"
EXTRA_PROMPTS = ["About the moon.", "About the computer."]  # never in any training data


def follows_format(answer):
    """FORMAT: one short sentence that starts with "The " and ends with a full stop, like our examples."""
    return answer.startswith("The ") and answer.endswith(".") and len(answer.split()) <= 8


def on_topic(prompt, answer):
    """CONTENT: does the answer actually talk about the thing asked about?"""
    subject = prompt.removeprefix("About the ").rstrip(".")
    return subject.lower() in answer.lower()


def show_answers(model, tok, prompts, label):
    print(f"\n{label}")
    fmt = topic = 0
    for p in prompts:
        answer = first_line(generate(model, tok, p + "\n", 20, temperature=0.0))
        fmt += follows_format(answer)
        topic += on_topic(p, answer)
        print(f"  {p:<22} -> {answer!r}")
    return fmt, topic


def main(argv=None):
    parser = argparse.ArgumentParser(description="Instruction-tune the story model.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--from-scratch", action="store_true", help="start from random weights instead")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if not args.pretrained.exists():
        parser.error(f"no pretrained model at {args.pretrained}. Run phase4_transformer\\train.py first.")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model, tok, info = load_checkpoint(args.pretrained)
    base, _, _ = load_checkpoint(args.pretrained)          # an untouched copy, for comparison
    if args.from_scratch:
        model = TinyGPT(model.cfg)                          # same shape and tokenizer, random weights
    ctx = model.cfg.context_length

    pairs = read_blocks(DATA / "instructions.txt", 2)
    train_pairs, held_out = split_every(pairs, 6)
    try:
        train_ex = [encode_pair(tok, p, r, ctx) for p, r in train_pairs]
        held_ex = [encode_pair(tok, p, r, ctx) for p, r in held_out]
    except ValueError as error:
        parser.error(str(error))

    # To measure forgetting: loss on the validation STORIES, which the pretrained model was judged on.
    _, val_stories = story_data.split_stories(story_data.load_stories(), 3)
    story_ids = torch.tensor(tok.encode(story_data.SEPARATOR.join(val_stories)))

    start = "random weights" if args.from_scratch else "the pretrained story model"
    print(f"Fine-tuning from {start}: {len(train_ex)} training pairs, {len(held_ex)} held out")
    print(f"Held-out prompts: {', '.join(p for p, _ in held_out)}")
    print(f"Story validation loss before fine-tuning: {story_train.evaluate(base, story_ids):.3f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"\n{'step':>5} {'train answers':>14} {'held-out answers':>17} {'stories (forgetting)':>21}")
    best = float("inf")
    model.train()
    for step in range(1, args.steps + 1):
        xs, ys = pad_batch(rng.sample(train_ex, min(args.batch, len(train_ex))))
        _, loss = model(xs, ys)    # ys has -100 on prompt tokens and padding, so only answers count
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0:
            held = response_loss(model, held_ex)
            note = ""
            if held < best:
                best = held
                save_checkpoint(args.out, model, tok, dict(info, step=step, held_out_loss=held,
                                                           training_text=info["training_text"] + "\n\n" +
                                                           "\n\n".join(f"{p}\n{r}" for p, r in train_pairs)))
                note = "  <- best, saved"
            print(f"{step:>5} {response_loss(model, train_ex):>14.3f} {held:>17.3f} "
                  f"{story_train.evaluate(model, story_ids):>21.3f}{note}")

    tuned, _, saved = load_checkpoint(args.out)
    print(f"\nSaved step {saved['step']} to {args.out}")
    print(f"Held-out answer loss: base model {response_loss(base, held_ex):.3f} -> fine-tuned {best:.3f}")
    print(f"Story validation loss: base {story_train.evaluate(base, story_ids):.3f} -> fine-tuned "
          f"{story_train.evaluate(tuned, story_ids):.3f}  (higher = it forgot some story-writing)")

    prompts = [p for p, _ in held_out]
    base_fmt, base_topic = show_answers(base, tok, prompts, "BASE MODEL on held-out instructions (greedy):")
    fmt, topic = show_answers(tuned, tok, prompts, "FINE-TUNED MODEL on held-out instructions (greedy):")
    n = len(prompts)
    print(f"\n  Right FORMAT (one short 'The ... .' sentence): base {base_fmt}/{n}, fine-tuned {fmt}/{n}")
    print(f"  On TOPIC (mentions what was asked about):     base {base_topic}/{n}, fine-tuned {topic}/{n}")
    print("Compare with the true answers:")
    for p, r in held_out:
        print(f"  {p:<22} -> {r!r}")
    show_answers(tuned, tok, EXTRA_PROMPTS, "FINE-TUNED MODEL on things that were never in any training data:")
    print("  It answers confidently in the learned style. Fine-tuning taught a FORMAT, not facts.")
    return {"held_out_loss": best, "format": fmt, "on_topic": topic, "base_format": base_fmt}


if __name__ == "__main__":
    main()
