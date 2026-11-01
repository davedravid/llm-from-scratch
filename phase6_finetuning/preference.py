"""Phase 6a: preference tuning with DPO (Direct Preference Optimisation), written by us.

Human feedback, in miniature. For each instruction we have two answers, one CHOSEN and one
REJECTED (in real life, people pick; here we wrote the labels ourselves: cheerful beats gloomy).
DPO nudges the model so the chosen answer becomes more likely and the rejected one less likely,
compared with a frozen copy of the starting model, so it can't drift too far.

Real RLHF usually trains a separate reward model on the human rankings and then optimises the
LLM against it with reinforcement learning. DPO skips the reward model and learns from the pairs
directly. It's a simpler, widely used alternative, and small enough to read in full.

Run (from the project folder, after finetune.py):
    .venv\\Scripts\\python phase6_finetuning\\preference.py
Options: --steps 60  --lr 0.0005  --beta 0.5  --seed 0
"""
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from common import (CHECKPOINTS, DATA, encode_pair, first_line, generate, load_checkpoint, read_blocks,
                    response_logprob, response_loss, save_checkpoint, split_every)
from finetune import OUT as FINETUNED

OUT = CHECKPOINTS / "preference.pt"


@torch.no_grad()
def preference_accuracy(model, chosen, rejected):
    """How often the model gives the chosen answer a higher probability than the rejected one.
    Beware: a longer answer has more tokens, and every token multiplies in a probability below 1,
    so longer answers start at a disadvantage (here " happy" is 5 tokens and " sad" is 2)."""
    was_training = model.training
    model.eval()
    gap = response_logprob(model, chosen) - response_logprob(model, rejected)
    model.train(was_training)
    return (gap > 0).float().mean().item(), gap.mean().item()


def mood_rates(model, tok, prompts, samples=5):
    """Share of sampled answers that contain 'happy' / 'sad'."""
    answers = [first_line(generate(model, tok, p + "\n", 15, 0.8, 0.9, seed=s)) for p in prompts for s in range(samples)]
    words = [a.lower().split() for a in answers]
    happy = sum(any(w.startswith("happy") for w in ws) for ws in words) / len(answers)
    sad = sum(any(w.startswith("sad") for w in ws) for ws in words) / len(answers)
    return happy, sad


def dpo_loss(policy, reference, chosen, rejected, beta):
    """-log sigmoid( beta * (how much MORE the policy likes chosen than the reference does
                              - how much MORE the policy likes rejected than the reference does) )"""
    with torch.no_grad():
        ref_chosen, ref_rejected = response_logprob(reference, chosen), response_logprob(reference, rejected)
    pol_chosen, pol_rejected = response_logprob(policy, chosen), response_logprob(policy, rejected)
    margin = beta * ((pol_chosen - ref_chosen) - (pol_rejected - ref_rejected))
    return -F.logsigmoid(margin).mean()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preference-tune the fine-tuned model with DPO.")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=0.5, help="how strongly to stay close to the starting model")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=Path, default=FINETUNED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if not args.start.exists():
        parser.error(f"no fine-tuned model at {args.start}. Run phase6_finetuning\\finetune.py first.")
    if args.beta <= 0:
        parser.error("--beta must be positive")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    policy, tok, info = load_checkpoint(args.start)
    reference, _, _ = load_checkpoint(args.start)     # frozen copy: the "don't drift too far" anchor
    ctx = policy.cfg.context_length

    pairs = read_blocks(DATA / "preferences.txt", 3)
    train_pairs, held_pairs = split_every(pairs, 5)
    enc = lambda ps, i: [encode_pair(tok, p[0], p[i], ctx) for p in ps]  # noqa: E731
    train_c, train_r, held_c, held_r = enc(train_pairs, 1), enc(train_pairs, 2), enc(held_pairs, 1), enc(held_pairs, 2)

    # Side effects to watch: does it still answer the instruction data as well?
    instructions = read_blocks(DATA / "instructions.txt", 2)
    instr_ex = [encode_pair(tok, p, r, ctx) for p, r in instructions]
    all_prompts = [p for p, _ in instructions]

    print(f"Preference pairs: {len(train_pairs)} for training, {len(held_pairs)} held out "
          f"({', '.join(p[0] for p in held_pairs)})")
    acc0, gap0 = preference_accuracy(policy, held_c, held_r)
    happy0, sad0 = mood_rates(policy, tok, all_prompts)
    instr0 = response_loss(policy, instr_ex)
    print(f"Before: held-out preference accuracy {acc0:.0%} (average log-prob gap {gap0:+.2f})")

    print(f"\n{'step':>5} {'DPO loss':>9} {'held-out accuracy':>18} {'avg gap':>8}")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    policy.train()
    for step in range(1, args.steps + 1):
        idx = rng.sample(range(len(train_c)), min(args.batch, len(train_c)))
        loss = dpo_loss(policy, reference, [train_c[i] for i in idx], [train_r[i] for i in idx], args.beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 10 == 0:
            acc, gap = preference_accuracy(policy, held_c, held_r)
            print(f"{step:>5} {loss.item():>9.3f} {acc:>18.0%} {gap:>+8.2f}")

    policy.eval()
    save_checkpoint(args.out, policy, tok, dict(info, preference_steps=args.steps, beta=args.beta))
    acc1, gap1 = preference_accuracy(policy, held_c, held_r)
    happy1, sad1 = mood_rates(policy, tok, all_prompts)
    instr1 = response_loss(policy, instr_ex)
    print(f"\nSaved to {args.out}")
    print(f"Held-out preference accuracy: {acc0:.0%} -> {acc1:.0%}   (gap {gap0:+.2f} -> {gap1:+.2f})")
    print(f"Answers mentioning 'happy' (all 36 prompts x 5 samples): {happy0:.0%} -> {happy1:.0%}")
    print(f"Answers mentioning 'sad':                                {sad0:.0%} -> {sad1:.0%}")
    print(f"Loss on the instruction answers (side effect):           {instr0:.3f} -> {instr1:.3f}")

    print("\nGreedy answers before -> after:")
    for p in ["About the dragon.", "About the witch.", "About the owl.", "About the sea."]:
        before = first_line(generate(reference, tok, p + "\n", 15, 0.0))
        after = first_line(generate(policy, tok, p + "\n", 15, 0.0))
        print(f"  {p:<20} {before!r:<34} -> {after!r}")
    print("Preference accuracy only scores the two FIXED answers in the data. It can reach 100% while the")
    print("model's own answers barely improve, or turn into fragments of the rewarded word ('rapy'): a")
    print("metric that improves isn't the same as behaviour that improves. And nothing in the preference")
    print("data is about truth: the stories say the dragon was lonely, but the data rewards 'happy'.")
    return {"accuracy": (acc0, acc1), "happy": (happy0, happy1), "sad": (sad0, sad1), "instr_loss": (instr0, instr1)}


if __name__ == "__main__":
    main()
