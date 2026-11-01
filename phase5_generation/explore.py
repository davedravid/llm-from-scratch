"""Phase 5a: explore logits, softmax, temperature, top-p and the context window
using the model trained in Phase 4b.

Run (from the project folder):
    .venv\\Scripts\\python phase5_generation\\explore.py
Just generate with your own settings:
    .venv\\Scripts\\python phase5_generation\\explore.py --temperature 0.8 --top-p 0.9
Options: --prompt "..."  --samples 10  --tokens 40  --checkpoint path
"""
import argparse
from pathlib import Path

import torch

import metrics
from sampling import (DEFAULT_CHECKPOINT, bpe, fit_to_context, generate, load_checkpoint,
                      next_token_probs, softmax_with_temperature)

SETTINGS = [  # (name, temperature, top_p)
    ("greedy", 0.0, 1.0),
    ("cautious", 0.5, 1.0),
    ("balanced", 0.8, 0.9),
    ("plain sampling", 1.0, 1.0),
    ("hot", 1.5, 1.0),
]


def show(token):
    return bpe.show_token(token)


def section_probabilities(model, tok, prompt):
    ids = torch.tensor([tok.encode(prompt)])
    with torch.no_grad():
        logits = model(ids)[0][0, -1]
    order = logits.argsort(descending=True)[:8].tolist()

    print(f"1. LOGITS -> PROBABILITIES for the token after {prompt!r}")
    print("   The model outputs one raw score (logit) per vocabulary token. Softmax turns them into")
    print("   probabilities that add up to 1. Top 8 of", len(tok.vocab), "tokens:\n")
    temps = [0.5, 1.0, 1.5]
    print(f"   {'token':<10}{'logit':>7}" + "".join(f"{'T=' + str(t):>9}" for t in temps))
    probs = {t: softmax_with_temperature(logits, t) for t in temps}
    for i in order:
        print(f"   {show(tok.vocab[i]):<10}{logits[i].item():>7.2f}"
              + "".join(f"{probs[t][i].item():>9.1%}" for t in temps))
    print("\n2. TEMPERATURE reshapes the same scores: T=0.5 makes the favourite more certain, "
          "T=1.5 spreads the chances out.")
    print(f"   Probability of the top token: " + ", ".join(f"T={t}: {probs[t][order[0]].item():.0%}" for t in temps))

    print("\n3. TOP-P keeps the smallest set of top tokens whose probabilities add up to at least p (at T=1):")
    for p in [0.2, 0.5, 0.9, 1.0]:
        kept = (next_token_probs(logits, 1.0, p) > 0).sum().item()
        print(f"   top_p={p:<4} keeps {kept:>3} of {len(tok.vocab)} tokens")
    print("   Not 'the top p% of tokens': the count depends on how confident the model is.\n")


def section_compare(model, tok, prompt, training_text, samples, tokens):
    known = set(metrics.words(training_text))
    print(f"4. SAME PROMPT, DIFFERENT SETTINGS ({samples} samples each, {tokens} tokens, seeds 0-{samples - 1})")
    print(f"   {'setting':<16}{'temp':>5}{'top_p':>6}{'distinct':>9}{'real words':>11}{'repeats':>8}{'copy':>5}")
    examples = []
    for name, t, p in SETTINGS:
        texts = [generate(model, tok, prompt, tokens, t, p, seed=s) for s in range(samples)]
        real = sum(metrics.real_word_rate(x, known) for x in texts) / samples
        rep = sum(metrics.repetition_rate(x) for x in texts) / samples
        copy = max(metrics.longest_copy(x, training_text) for x in texts)
        print(f"   {name:<16}{t:>5}{p:>6}{metrics.distinct_outputs(texts):>6}/{samples:<2}"
              f"{real:>11.0%}{rep:>8.0%}{copy:>5}")
        examples.append((name, texts[0]))
    print("   distinct = how many samples differ; real words = % found in the training text;")
    print("   repeats = % of 3-word runs already used in the same sample; copy = longest word-for-word copy\n")
    print("   First sample for each setting:")
    for name, text in examples:
        print(f"   [{name}] {prompt}{text!r}")
    print()


def section_context(model, tok, training_text):
    limit = model.cfg.context_length
    long_prompt = training_text[:300]
    ids = tok.encode(long_prompt)
    kept, dropped = fit_to_context(ids, limit)
    print(f"5. THE CONTEXT WINDOW: this model sees at most {limit} tokens")
    print(f"   A {len(ids)}-token prompt. Only the last {limit} are used; {len(dropped)} fall off the front:")
    print(f"   DROPPED: {tok.decode(dropped)!r}")
    print(f"   KEPT   : {tok.decode(kept)!r}")
    try:
        model(torch.tensor([ids]))
    except ValueError as error:
        print(f"   Feeding all {len(ids)} tokens straight to the model fails: {error}")
        print("   So the generation code has to crop. That's an application decision, not the model's.")

    # Swap the dropped part for a completely different story and generate from both prompts.
    other = training_text[2000:2000 + len(tok.decode(dropped)) + 50]
    swapped_ids = tok.encode(other)[:len(dropped)] + kept
    original = generate(model, tok, tok.decode(ids), 20, 0.8, 0.9, seed=0)
    swapped = generate(model, tok, tok.decode(swapped_ids), 20, 0.8, 0.9, seed=0)
    print(f"   Now replace the dropped part with a different story and generate (same seed):")
    print(f"     original prompt -> {original!r}")
    print(f"     swapped prompt  -> {swapped!r}")
    print(f"   {'Identical' if original == swapped else 'Different'}: what fell out of the window has no effect.")
    print("   If instructions or key facts were at the start of that prompt, the model would never see them.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explore sampling and the context window.")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, help="just generate with this temperature")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args(argv)
    if not args.checkpoint.exists():
        parser.error(f"no trained model at {args.checkpoint}. Run phase4_transformer\\train.py first.")
    if args.samples < 1 or args.tokens < 1:
        parser.error("--samples and --tokens must be at least 1")
    model, tok, info = load_checkpoint(args.checkpoint)
    if not tok.encode(args.prompt):
        parser.error("--prompt is empty")

    if args.temperature is not None:
        try:
            for seed in range(args.samples):
                text = generate(model, tok, args.prompt, args.tokens, args.temperature, args.top_p, seed)
                print(f"[seed {seed}] {args.prompt}{text}")
        except ValueError as error:
            parser.error(str(error))
        return

    section_probabilities(model, tok, args.prompt)
    section_compare(model, tok, args.prompt, info["training_text"], args.samples, args.tokens)
    section_context(model, tok, info["training_text"])


if __name__ == "__main__":
    main()
