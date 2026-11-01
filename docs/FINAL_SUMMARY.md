# Final summary

## What was delivered

| Deliverable | Where |
|---|---|
| Working, documented project | `README.md`, `PROGRESS.md` |
| A web page to try everything | `app/` (run `app\server.py`) |
| Small runnable examples | one folder per phase, `phase1_chatbot` to `phase6b_tools` |
| Trained toy transformer | `phase4_transformer/train.py`, `generate.py` |
| Saved weights, and what they contain | `phase4_transformer/checkpoints/tinygpt.pt`; `docs/ARCHITECTURE.md` |
| Training and validation results, with limitations | `PROGRESS.md`, Phase 4b |
| Evaluation suite (normal, edge-case, adversarial) | `phase5b_evaluation/` |
| Guardrails and their known weaknesses | `phase5b_evaluation/README.md` |
| Tool-calling demonstration with validation | `phase6b_tools/` |
| Glossary | `GLOSSARY.md` |
| Architecture diagram and pipeline | `docs/ARCHITECTURE.md` |
| Where the video simplifies or is wrong | `docs/video-notes.md` |

## What the experiments showed

| Claim | Evidence from this project | Phase |
|---|---|---|
| Learned behaviour can come from data, not rules | XOR: 1 layer 2/4, 2 layers 4/4 | 2 |
| A model is a file of numbers plus code that uses them | 17 numbers reproduce XOR; 35,469 for the story model | 2, 4 |
| Tokens set cost and context use | Same length, 9 tokens vs 23 tokens | 3a |
| Embeddings capture usage, not meaning | happy~sad +0.88 | 3b |
| Lower training loss can mean a worse model | Training loss 0.13, validation 10.6, worse than guessing | 4b |
| Fluent output can be memorised | A 48-word verbatim copy of a training story | 4b |
| Sampling trades variety against errors | Real words: 82% greedy, 53% at T=1.5 | 5a |
| Text outside the window has no effect | Swapped 82 dropped tokens: identical output | 5a |
| A prompt rule didn't control this model | "never" 10/100, "always" 11/100, no rule 13/100 | 5b |
| Repetition drives leaks | Fake secret: 0/21 seen once, 21/21 seen 5 times (overfitted) | 5b |
| Fine-tuning taught format, not facts | Format 6/6, on topic 0/6 | 6a |
| Pretraining helps fine-tuning | Held-out loss 2.16 vs 3.31 from scratch | 6a |
| A score can improve while behaviour doesn't | Preference accuracy 100%, answers like "The rapy." | 6a |
| Permissions hold even when the model is fooled | Injected delete was proposed, and refused | 6b |

## What this project can't show

These are toy models on tiny data we mostly wrote ourselves. They don't show how large models behave,
and the small test sets and single runs mean the exact numbers are noisy. The prompt-rule result applies to our
untuned model: instruction-tuned models do follow prompts much of the time. **Nothing here establishes
production-level safety.** The fine-tuning and tool models also use hand-picked formats that suit them.

## Check yourself: can you explain these without notes?

The project was confirmed phase by phase, but the check questions were never answered in the conversation, so
whether these are solid is yours to judge. If one feels shaky, that's the place to go back to.

1. How a chatbot built on rules differs from an LLM.
2. What weights, loss and backpropagation are, and how they fit together.
3. Why a model needs tokens and embeddings, and why token count affects cost.
4. What attention does, and why the causal mask is needed.
5. How one token is chosen, and how a whole answer is built from that.
6. What temperature and top-p do, and how top-p differs from "the top p% of tokens".
7. Why validation data is needed, and what overfitting looks like.
8. How pretraining, fine-tuning and preference tuning differ, and what each one changes.
9. Why LLMs hallucinate, and why fine-tuning doesn't fix it.
10. Which controls belong in the model, the prompt, the application and evaluation.
11. Why a model needs external software to take an action, and what the execution boundary is.

## Questions to ask about any AI feature

- **Capability:** what was it tested on, and does that look like our real inputs?
- **Limitations:** what happens with long inputs, unfamiliar topics or other languages?
- **Cost:** how many tokens does a typical request use, including history and tool results?
- **Performance:** how does the loop's length affect response time?
- **Quality:** which metric improved, on which held-out data, and did real outputs improve as well?
- **Risk:** what's the worst action it can take if it's fooled, and which code stops that?
- **Evaluation:** who wrote the test cases, what are the false-positive and false-negative rates, and what's monitored after launch?

## What to learn next

1. **Retrieval (RAG):** the standard way to give a model facts. It builds directly on embeddings and the context window.
2. **Evaluating real LLM products:** labelled test sets, human review, and measuring guardrail error rates.
3. **A real model through an API:** tool calling, prompt caching and token costs, compared with what you built here.
4. **Scaling:** what changes with GPUs and large datasets, by training on a bigger public dataset.
5. **Interpretability:** what research has found inside real models, such as the previous-token heads you saw in Phase 4b.
