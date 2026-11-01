"""Phase 6b: train a small transformer to PROPOSE tool calls, and measure it on unseen requests.

Why a new model instead of fine-tuning Phase 6a's? That tokenizer has no digits, braces or quotes, so it
literally can't write a tool call, and its 32-token window is too short for a whole conversation.
Same TinyGPT code; new character-level tokenizer, 144-character window, generated data.

Run (from the project folder, takes a few minutes on CPU):
    .venv\\Scripts\\python phase6b_tools\\tool_model.py
Options: --steps 3000  --dim 64  --layers 3  --seed 0
"""
import argparse
import sys
import time
from pathlib import Path

import torch

import harness
import tool_data
import tools

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "phase4_transformer"))
sys.path.insert(0, str(HERE.parent / "phase6_finetuning"))

from common import pad_batch  # noqa: E402  (Phase 6a)
from generate import load_checkpoint, save_checkpoint  # noqa: E402  (Phase 4b)
from model import Config, TinyGPT  # noqa: E402
from tokens import bpe  # noqa: E402

CHECKPOINT = HERE / "checkpoints" / "tool_model.pt"
MAX_LINE = 80


class TrainedModel:
    """Makes the transformer look like what the harness expects: text in, one line out."""

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

    @torch.no_grad()
    def __call__(self, transcript):
        ids = self.tokenizer.encode(transcript)
        newline = self.tokenizer.id_of["\n"]
        out = []
        for _ in range(MAX_LINE):
            window = (ids + out)[-self.model.cfg.context_length:]
            next_id = self.model(torch.tensor([window]))[0][0, -1].argmax().item()
            if next_id == newline:
                break
            out.append(next_id)
        return self.tokenizer.decode(out)


def load_trained(path=CHECKPOINT):
    model, tokenizer, _ = load_checkpoint(path)
    return TrainedModel(model, tokenizer)


def score(trained, cases=None):
    """Run unseen requests through the real harness. Returns per-kind and overall results."""
    cases = cases or tool_data.held_out_requests()
    rows = []
    for request, expected_call, expected_answer in cases:
        runner = tools.ToolRunner()
        outcome = harness.run_request(trained, request, runner)
        proposal = outcome.log[0]["proposal"] if outcome.log else ""
        rows.append({"request": request, "kind": request.split()[0], "proposal": proposal, "answer": outcome.answer,
                     "call_ok": proposal == expected_call, "answer_ok": outcome.answer == expected_answer,
                     "expected_answer": expected_answer, "status": outcome.status})
    return rows


def summarise(rows):
    kinds = {}
    for r in rows:
        k = kinds.setdefault(r["kind"], [0, 0, 0])
        k[0] += 1
        k[1] += r["call_ok"]
        k[2] += r["answer_ok"]
    return kinds


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the tool-calling model.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=CHECKPOINT)
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    torch.manual_seed(args.seed)
    tokenizer = bpe.train(tool_data.CHARSET, num_merges=0)    # one token per character
    cfg = Config(vocab_size=len(tokenizer.vocab), context_length=tool_data.CONTEXT, dim=args.dim, heads=4, layers=args.layers)
    model = TinyGPT(cfg)
    print(f"Model: {cfg.layers} blocks, {cfg.dim} dims, {cfg.vocab_size} tokens, {cfg.context_length}-character "
          f"window = {model.count_parameters():,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trained = TrainedModel(model, tokenizer)
    started = time.time()
    report_every = max(1, args.steps // 6)
    for step in range(1, args.steps + 1):
        xs, ys = pad_batch(tool_data.batch(tokenizer, args.batch, seed=args.seed * 100_000 + step))
        _, loss = model(xs, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % report_every == 0 or step == args.steps:
            model.eval()
            rows = score(trained)
            model.train()
            correct = sum(r["answer_ok"] for r in rows)
            print(f"  step {step:>5}  loss {loss.item():.3f}  unseen requests fully correct "
                  f"{correct}/{len(rows)}  ({time.time() - started:.0f}s)", flush=True)

    # Keep the last step. The data is fresh every batch, so there's nothing to overfit to in the usual way,
    # and picking the "best" step using these same unseen requests would flatter the final score.
    model.eval()
    save_checkpoint(args.out, model, tokenizer, {"step": args.steps, "settings": {k: str(v) for k, v in vars(args).items()}})
    rows = score(load_trained(args.out))
    print("\nOn requests never used in training (through the real harness, read-only permissions):")
    print(f"  {'kind':<8}{'total':>6}{'right call':>12}{'right answer':>14}")
    for kind, (n, calls, answers) in summarise(rows).items():
        print(f"  {kind:<8}{n:>6}{calls:>12}{answers:>14}")
    wrong = [r for r in rows if not r["answer_ok"]]
    print(f"\nWrong ({len(wrong)}):")
    for r in wrong:
        print(f"  {r['request']:<20} proposed {r['proposal']!r}, answered {r['answer']!r} "
              f"(expected {r['expected_answer']!r})")
    return rows


if __name__ == "__main__":
    main()
