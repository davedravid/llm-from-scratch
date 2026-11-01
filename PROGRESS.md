# Progress Log

## Status

| Phase | Status |
|---|---|
| 0 Planning and setup | Done (approved 2026-10-29) |
| 1 Rule-based chatbot | Done (confirmed 2026-10-29) |
| 2 Neural networks (XOR) | Done (confirmed 2026-10-29) |
| 3a Tokenization (BPE) | Done (confirmed 2026-10-29) |
| 3b Embeddings (word2vec) | Done (confirmed 2026-10-29) |
| 4a Transformer architecture | Done (confirmed 2026-10-29) |
| 4b Training the transformer | Done (confirmed 2026-10-29) |
| 5a Sampling and the context window | Done (confirmed 2026-10-29) |
| 5b Evaluation suite and guardrails | Done (confirmed 2026-10-29) |
| 6a Fine-tuning and preference tuning | Done (confirmed 2026-11-01) |
| 6b Tool calling, and the final architecture write-up | Done (2026-11-01) |
| App: joined pipeline and web page | Done (2026-11-01) |
| Review of your understanding | Open: you chose to work backward from the app |

## Phase 0: Planning and setup (2026-10-29)

**Environment found:** Windows 11, Python 3.11.9, pip 24.0, git 2.56. AMD Ryzen AI 7 350 (16 threads),
about 23 GB RAM, AMD Radeon 860M integrated graphics. No NVIDIA GPU, so we train on the CPU. numpy and torch
aren't installed yet.

**Source material:** transcript only. The video's code isn't available, so we write our own equivalent.

**Created:** README, this log, glossary, `check_env.py`, `requirements.txt`,
`docs/video-notes.md`.

**Known issues:** none yet.

**Next step:** after approval, Phase 1 (rule-based chatbot, standard library only).

## Phase 1: Rule-based chatbot (2026-10-29)

**Built:** `phase1_chatbot/chatbot.py` (3 rules + fallback, input validation, demo mode) and
`phase1_chatbot/test_chatbot.py` (16 tests, all passing).

**Concepts:** rule-based systems, the black-box interface (`respond(text) -> text`), normalisation,
rule order, application-level validation kept separate from the bot.

**Experiments:**
- Reordering the rules (feeling before greeting) changes "I feel tired, hi there" from a greeting to
  "Why do you feel tired hi there?", but does *not* change "Hey, I feel sad". The feeling rule only
  matches messages that start with "I feel", so reordering alone can't fix it.

**Known limitations (documented as tests):** the greeting hides the real question, negation is
ignored, the "my" rule picks the wrong subject, the first matching rule wins, and the bot can't follow
any instruction (so it also can't be hijacked).

**Next step:** Phase 2, neural networks with XOR (needs numpy installed in a virtual environment).

## Phase 2: Neural networks, XOR (2026-10-29)

**Setup issue fixed:** creating the virtual environment failed. The session's folder path is 180 characters,
and pip's files went over Windows' 260-character path limit (long paths are disabled on this machine).
Fix: moved the project to a short folder path, a short path outside OneDrive.
numpy 2.4.6 is installed in `.venv`.

**Built:** `phase2_neural_net/xor.py` (single- and multi-layer networks, hand-written backpropagation,
weights saved as JSON) and `phase2_neural_net/test_xor.py` (10 tests, all passing).

**Bug found and fixed:** the "before training" table had already taken one training step. Init and
training are now separate functions.

**Concepts:** neurons, weights, bias, parameters, sigmoid, loss (mean squared error), epochs,
learning rate, backpropagation, and a model being just a file of numbers.

**Experiments (seed 0, 5000 epochs):**
| Setup | Result |
|---|---|
| Single layer, XOR | 2/4 correct, loss stuck at 0.25, all weights about 0 (outputs 0.5 for everything) |
| Single layer, AND / OR | 4/4 each |
| Multi, 4 hidden | 4/4, loss 0.0003 (17 parameters) |
| Multi, 1 hidden (seeds 0-4) | 3/4 every time, loss about 0.167 |
| Multi, 2 or 3 hidden (seeds 0-4) | 4/4 every time |
| Learning rate 0 / 0.1 / 10 / 100 | nothing learned / 4/4 but slower (loss 0.023) / 4/4 fast / fails (loss 0.5) |

**Known limitations:** we test on the same 4 examples we trained on (there's nothing else for XOR),
so this phase can't show generalisation. That comes with real data in Phases 3-4. Results depend on
the random starting weights (the seed).

**Next step:** Phase 3, tokenization (BPE) and embeddings.

## Phase 3a: Tokenization with BPE (2026-10-29)

**Split:** Phase 3 is now 3a (tokenizer) and 3b (embeddings), to keep each step small.

**Built:** `data/stories.txt` (30 stories we wrote, 7,259 characters), `phase3a_tokenizer/bpe.py`
(character-level BPE: training, encode, decode, save/load) and `phase3a_tokenizer/test_bpe.py`
(15 tests, all passing).

**Bug found and fixed:** two tests crashed because `main()` reconfigured the screen output, which the
tests replace. Moved that line so it only runs when the script is run directly.

**Experiments ("Once upon a time there was a king." = 34 characters):**
| Merges | Vocabulary | Tokens for that sentence | Whole training file |
|---|---|---|---|
| 0 | 41 | 34 (one per character) | 7,259 tokens |
| 10 | 51 | 25 | 5,867 |
| 50 | 91 | 17 | 4,279 |
| 200 | 241 | 9 (one per word) | 2,696 |
| 467 (default stops here, min-count 2) | 508 | 9 | 1,918 |
| 723 (min-count 1) | 764 | 9 | 1,662 |

Unfamiliar text costs more: "The astronaut zoomed past Jupiter." is 5 words but 23 tokens. French
is 6 words, 16 tokens.

**Known limitations:** character-level, so unseen characters (capital J, emoji) become `<unk>` and are
lost on decode. Real tokenizers work on bytes so nothing is unknown. The vocabulary only reflects our
tiny English story world.

**Next step:** Phase 3b, embeddings (word2vec skip-gram with negative sampling).

## Phase 3b: Embeddings with word2vec (2026-10-29)

**Built:** `data/relations.txt` (98 sentences we wrote, 156 unique words), `phase3b_embeddings/word2vec.py`
(skip-gram with negative sampling, cosine similarity, nearest neighbours, analogies, save/load) and
`phase3b_embeddings/test_word2vec.py` (14 tests, all passing). It trains in about 2 seconds.

**Fixed while building:** an unknown word in `--analogy` was only reported after training. It's now
rejected immediately.

**Default results (16 dims, window 2, 300 epochs):** king~queen +0.85, happy~joyful +0.99,
happy~refrigerator +0.11 (before training: -0.59, +0.01, -0.09). All 5 analogies correct.

**Experiments (analogies solved out of 5):**
| Setting | Result |
|---|---|
| Seeds 0-9 | 5/5 every time |
| Dimensions 2 / 4 / 8 / 32 / 64 (seeds 0-2) | 0,0,0 / 1,1,3 / 5,5,2 / 5,5,5 / 5,5,5 |
| Window 1 / 3 / 5 | 5/5 for all |
| Epochs 5 / 20 / 50 | 0,0,0 / 5,4,5 / 5,5,5. At 5 epochs everything is about 0.95 similar to everything |
| Not excluding the input words | king - man + woman gives "king" first; 2 of 3 analogies fail |

**Known limitations:**
- Opposites end up close (happy~sad +0.88): embeddings capture *usage*, not meaning.
- Unrelated words can still score high (king~banana +0.61), because a tiny corpus gives few contrasts.
- Analogies work because we **designed** the corpus with matched sentences, and we only test 5.
- The loss plateaus at about 1.6 from epoch 50. It can't reach 0 because the negative words are random.
- Whole words only: an unseen word has no vector at all.

**Next step:** Phase 4, the transformer (needs torch installed).

## Phase 4a: Transformer architecture (2026-10-29)

**Setup:** installed PyTorch 2.15.0 (CPU build) into `.venv`. It uses 8 threads.

**Split:** Phase 4 is now 4a (architecture, untrained) and 4b (training).

**Built:**
- `phase4_transformer/attention_by_hand.py`: one attention step with hand-picked numbers. "bank" leans
  nature (0.67 vs 0.22) after "river" and money (0.22 vs 0.67) after "money".
- `phase4_transformer/model.py`: TinyGPT, our own attention (Q/K/V, scaling, causal mask, 4 heads),
  feed-forward, residual connections and layer norm. Only PyTorch's basic building blocks are used.
- `phase4_transformer/tokens.py`: loads the Phase 3a tokenizer.
- `phase4_transformer/inspect_model.py`: follows one sentence through the untrained model.
- `phase4_transformer/test_model.py`: 12 tests, all passing.

**Observed (untrained, default 2 layers, 32 dims, 4 heads, 508-token vocabulary):**
- 59,324 parameters. 56% are in the token embedding and output layer (which scale with vocabulary size);
  the attention layers are only 14%.
- Loss 6.52, against 6.23 for pure guessing (ln 508). Top predictions are junk at about 0.8% each.
- Attention is spread roughly evenly over earlier tokens, with exact zeros above the diagonal.
- Causal mask on: changing the last token changes only the last position. Off: all 8 positions change.

**Sizes:** 1 layer 46,716; 6 layers 109,756; dim 64 167,292; dim 128 with 8 heads 530,684 parameters.

**Known limitations:** untrained, so it says nothing about quality yet. The context window is fixed
at 32 tokens, and longer input is rejected. The position embedding is learned, so the model can't use
positions beyond 32 even in principle.

**Next step:** Phase 4b, training on `data/stories.txt` with a held-out validation set.

## Phase 4b: Training the transformer (2026-10-29)

**Built:** `data.py` (27/3 train/validation story split, random training windows), `train.py`
(AdamW training loop, deterministic evaluation, best-validation checkpointing, loss per character),
`generate.py` (checkpoint save/load, greedy generation, a word-for-word copy detector, attention
printout), dropout added to `model.py`, and `test_training.py` (16 tests; 28 in Phase 4 overall, all passing).

**Data hygiene:** the tokenizer is retrained on the 27 training stories only, so the validation stories
don't influence the vocabulary.

**Bugs found and fixed:**
- Per-token validation losses were being compared across tokenizers, which is unfair. Added loss per character.
- Saving the settings stored a Python `Path` object, which the safe checkpoint loader (`weights_only=True`)
  rejects. Stored it as text instead of loosening the security setting.

**Default run (100 merges = 141-token vocabulary, 35,469 parameters, dropout 0.2, 1000 steps, 21 seconds):**
best validation at step 350 (train 2.04, val 3.21, val per char 1.527; random guessing 4.95). After that,
train keeps falling (to 0.86) while val rises (to 4.01): overfitting.
Sample: "Once upon a time a litttle girl lived in the farm. The farmerily..." (longest copy: 3 words).

**Experiments (best validation loss per character, lower is better):**
| Setting | Val per char | Notes |
|---|---|---|
| 500 merges, no dropout (original) | 1.658 | best at step 100, val 10.6 by step 1500 (worse than guessing, 6.17) |
| 100 merges, dropout 0 / 0.1 / 0.2 / 0.3 | 1.536 / 1.519 / 1.527 / 1.509 | dropout barely matters here |
| 50 merges / 20 merges / 0 merges (characters) | 1.609 / 1.565 / 1.475 | smallest vocabulary generalises best on tiny data |
| dim 16 / 32 / 64 | 1.532 / 1.527 / 1.551 | bigger overfits sooner (best at step 150) |
| Learning rate 0.001 | 1.511 | slower but similar |
| Only 5 training stories | 1.950 | **data matters most** |
| Overfit (500 merges, no dropout, 1500 steps, keep last) | train 0.13 | writes perfect prose, a **48-word verbatim copy** of a training story |

**Attention learned:** block 1 head 2 has "was" put 0.92 of its attention on "there" (the previous token);
block 1 head 3 has "a" put 0.92 on "upon". These are previous-token heads.

**Known limitations:**
- About 3,000 training tokens is far too little to learn language. Output is locally plausible and
  globally meaningless, with invented words.
- Greedy decoding loops ("fouse fouse fouse").
- The validation set is 3 stories (370 tokens), so the numbers are noisy.
- The best step is chosen using the validation set, so its validation score is slightly optimistic.
  A third, untouched "test" set would fix that.

**Next step:** Phase 5, logits, softmax, temperature, top-p, context windows, evaluation and guardrails.

## Phase 5a: Sampling and the context window (2026-10-29)

**Split:** Phase 5 is now 5a (sampling, context) and 5b (evaluation suite, guardrails).

**Built:** `phase5_generation/sampling.py` (temperature, top-p, sampling, context cropping and generation,
all ours), `metrics.py` (real-word rate, repetition rate, longest copy, distinct outputs), `explore.py`
(a 5-section demo using the Phase 4b model) and `test_sampling.py` (22 tests, all passing).

**Bug found and fixed:** the first version of the context-window check compared the kept text with
itself, so it was "identical" by construction and proved nothing. Replaced it with a real test: swap the
dropped text for a different story and generate with the same seed. The output is still identical.

**Observed for "Once upon a time":** the top two next tokens are " a" (57%) and " there" (40%). Temperature
0.5 / 1.0 / 1.5 gives the top token 68% / 57% / 45%. Top-p 0.2 and 0.5 keep 1 token; 0.9 keeps 2;
1.0 keeps all 141.

**Settings comparison (10 samples, 40 tokens):**
| Setting | Distinct | Real words | Repeats | Longest copy |
|---|---|---|---|---|
| greedy (T=0) | 1/10 | 82% | 7% | 3 |
| T=0.5 | 10/10 | 72% | 1% | 5 |
| T=0.8, top-p 0.9 | 10/10 | 67% | 0% | 5 |
| T=1.0 | 10/10 | 67% | 0% | 5 |
| T=1.5 | 10/10 | 53% | 0% | 3 |
For "The dragon", greedy was 72% and T=0.8 / top-p 0.9 was 73%: the best setting depends on the prompt.

**Context window:** a 114-token prompt keeps only the last 32 tokens. Sending all 114 to the model raises
an error, so cropping is an application decision. Replacing the dropped 82 tokens with a different story
gives identical output.

**Known limitations:** the metrics are crude proxies (a "real word" can still be the wrong word), samples
are small (10 per setting), and none of this measures truth or usefulness.

**Next step:** Phase 5b, evaluation suite and guardrails.

## Phase 5b: Evaluation suite and guardrails (2026-10-29)

**Built:** `phase5b_evaluation/guardrails.py` (input checks, output checks, the guarded pipeline, a decision
log), `eval_cases.py` (29 cases in 10 categories), `run_eval.py` (four outcome types, a comparison
without guards, the prompt-rule experiment, saved results, a non-zero exit on FAIL), `canary.py`
(memorisation test), `test_guardrails.py` (17 tests) and `README.md` (the four layers and the
known-weaknesses record W1-W10). Added `--extra-train` to `phase4_transformer/train.py`.

**Results:** 22 PASS, 7 KNOWN WEAKNESS, 0 FAIL. All 122 tests across all phases pass.

**Corrections made during the phase (evaluation lessons):**
- E6 failed on the first run. My expectation was wrong, not the code ("!" is unseen, so blocking is right).
- H1 "passed" for the wrong reason (unfamiliar words). I replaced it with a familiar-word factual question;
  it's now a documented weakness.
- The prompt rule seemed to work with 20 samples (4 → 1). With 100 samples and controls: no rule 13,
  "never" 10, "always" 11, unrelated 13. No obedience.

**Canary:** a fake secret seen once never leaked. Seen 5 times, the overfitted model leaked it 21/21; the
early-stopped model 0/21. The output guard allowed every leak, because the secret is under the 12-word copy limit.

**Next step:** Phase 6, fine-tuning, human-feedback preference tuning and tool calling.

## Phase 6a: Fine-tuning and preference tuning (2026-10-29)

**Split:** Phase 6 is now 6a (fine-tuning, preference tuning) and 6b (tool calling, final write-up).

**Built:** `data/instructions.txt` (36 pairs), `data/preferences.txt` (26 pairs), `phase6_finetuning/common.py`
(prompt-masked examples, padding, response loss and log-probability), `finetune.py`, `preference.py` (DPO,
written by us) and `test_finetuning.py` (14 tests). All 136 tests across the project pass.

**Problems found and fixed:**
- **Context window too small for the first format.** "Tell me about the king." alone was 12 tokens, and the
  examples were 36-51 tokens against a limit of 33. Switched to "About the king." plus one-sentence answers.
- **A training-mode bug.** Phase 4's `evaluate` and our `response_loss` left models in training mode
  (dropout on), so "deterministic" generation was random (base on-topic varied 0/6 vs 1/6). Both now
  restore the previous mode, and there are regression tests.
- My first format check also required the right subject, which mixed up format and content. Split into two metrics.

**Fine-tuning (30 train / 6 held-out pairs, best at step 50, about 5 seconds):**
| | Base model | Fine-tuned from pretrained | Fine-tuned from scratch |
|---|---|---|---|
| Held-out answer loss | 4.848 | **2.162** | 3.309 |
| Right format (short "The ... ." sentence) | 0/6 | 6/6 | 6/6 |
| On topic (mentions the subject) | 0/6 | 0/6 | 0/6 |
| Story validation loss (forgetting) | 3.210 | 3.362 | 5.005 |
Example: "About the fox." gives "The roved told stary." (true answer "The fox was friendly.").
"About the moon." gives "The rold could talk.": a confident answer about something never in the data.

**Preference tuning (DPO, 21 train / 5 held-out pairs, beta 0.5, 60 steps, about 8 seconds):**
- Held-out preference accuracy 0% → 100% (log-prob gap -4.34 → +14.28).
- But "happy" appeared in 0% → 1% of answers. Greedy answers became "The rapy.", "The withapy.".
- The 0% starting point is a length bias: " happy" is 5 tokens, " sad" is 2.
- beta 0.1: 12% "happy", but instruction loss 1.49 → 1.86 and answers like "The rappppy." (drift).
  20 steps: 0% "happy". 200 steps at lr 0.001: 6%.

**Known limitations:** 30 instruction examples; the model can't copy the subject from the prompt;
preference labels were written by us (not humans) and only test a cheerful-vs-gloomy preference;
the held-out sets have 6 and 5 items.

**Next step:** Phase 6b, tool calling with validation, then the final architecture diagram and summary.

## Phase 6b: Tool calling and final write-up (2026-11-01)

**Built:** `phase6b_tools/` (`tools.py` validation and permissions, `harness.py` the loop, `tool_data.py` generated
conversations, `tool_model.py` training, `demo.py`, 21 tests, and a README with weaknesses T1-T7), plus
`docs/ARCHITECTURE.md` and `docs/FINAL_SUMMARY.md`.

**Results on 36 unseen requests:** right call 36/36, right final answer 31/36 (add 20/20, about 6/10, delete 5/6).
All 5 failures were garbled answers after a correct call. All 13 scripted bad proposals and the endless loop were stopped.

**Problems found:** the Phase 6a tokenizer can't write tool calls, so this is a new model. Mistyped real words in
the first run (right calls 24/36) were fixed with more varied training words. That caused a window overflow crash,
fixed by widening the window from 128 to 144.

**Known issues:** see T1-T7 in `phase6b_tools/README.md`. The guardrails and the tool harness aren't joined.
The check questions from each phase weren't answered in the conversation.

**Next step:** the app, below.

## App: joined pipeline and web page (2026-11-01)

**Built:** `app/pipeline.py` (input checks, model, tool harness, answer check and output checks for all four models),
`app/server.py` (local only, standard library), `app/index.html` and `app/test_app.py` (13 tests).
All 170 tests across 10 suites pass. Also checked in the browser, including phone width.

**Closed:** T1 (a garbled answer after a correct tool call is now flagged and replaced by the tool result) and T7
(the guardrails and tool harness are joined). The delete approval now shows the exact call before it runs.

**Problems found:** an empty prompt with guardrails off crashed the model (now a clear error), and the single-threaded
server hung on the browser's idle connections (now handled in parallel, with models used one request at a time).

**Next step:** work backward from the app through the concepts, at your pace.
