# Small LLM From Scratch: Learn by Building

A learning project that rebuilds every stage of a language model, from a rule-based
chatbot up to a tiny trained transformer with sampling, guardrails and tool calling.
It follows the structure of a YouTube video on building an LLM from scratch (the Syntax channel). See `JOURNEY.md` for how it went and `LIMITATIONS.md` for what it can't do.

**This is not a useful chatbot.** It's a teaching model, small enough to train on a laptop CPU
in minutes and simple enough to read every line of.

## What is ours and what is borrowed

| Component | Who wrote it |
|---|---|
| Chatbot, XOR network, BPE tokenizer, word embeddings, attention, transformer blocks, training loop, sampling, guardrails, tool harness | **Us** (this project) |
| Array maths (`numpy`) | External library |
| Tensors + automatic gradients + optimizer (`torch`, CPU only) | External library, used from Phase 4 |
| Pretrained models / Hugging Face / APIs | **Not used** |

The video's original code isn't available to us. Everything here is our own equivalent
implementation. Where the video simplifies something or gets it wrong, see `docs/video-notes.md`.

## Phases

| Phase | Topic | Video sections | Needs |
|---|---|---|---|
| 0 | Planning and setup | - | Python only |
| 1 | Rule-based chatbot (the "black box") | Eliza, chatbot code | Python only |
| 2 | Neural networks: XOR, weights, loss, backpropagation | Neural networks, XOR | numpy |
| 3 | Tokenization (BPE) and embeddings (word2vec) | Tokenization, embeddings | numpy |
| 4 | Transformer: attention, Q/K/V, feed-forward, training | Transformers | torch (CPU) |
| 5 | Logits, softmax, temperature, top-p, generation, context window, evaluation + guardrails | Sampling, autoregressive loop, context | torch (CPU) |
| 6 | Fine-tuning, human-feedback preference tuning, tool calling | Fine-tuning, RLHF, tools | torch (CPU) |

Each phase gets its own folder (`phase1_chatbot/`, `phase2_neural_net/`, ...) when we build it.

## Setup

You need Python 3.10 or newer (this machine has 3.11.9). Check your environment with:

```
python check_env.py
```

From Phase 2 onward the project uses a private Python environment in `.venv` (a "virtual environment":
a folder holding this project's libraries so they don't affect anything else on your computer).
It's already set up with numpy 2.4.6 and PyTorch 2.15.0 (CPU-only build). To recreate it from scratch:

```
python -m venv .venv
.venv\Scripts\python -m pip install numpy
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Run every command **from the project folder** (for example `C:\Projects\llm-from-scratch`), and start
each one with `.venv\Scripts\python` so it uses the project's libraries.

> **Windows path-length note:** Windows limits file paths to 260 characters unless long paths are
> enabled. Library installs fail inside deeply nested folders, which is why the project lives at a
> short path outside OneDrive.

### Running each phase

| Phase | Run | Test |
|---|---|---|
| 1 | `.venv\Scripts\python phase1_chatbot\chatbot.py --demo` | `.venv\Scripts\python -m unittest discover -s phase1_chatbot -v` |
| 2 | `.venv\Scripts\python phase2_neural_net\xor.py --layers multi` | `.venv\Scripts\python -m unittest discover -s phase2_neural_net -v` |
| 3a | `.venv\Scripts\python phase3a_tokenizer\bpe.py` | `.venv\Scripts\python -m unittest discover -s phase3a_tokenizer -v` |
| 3b | `.venv\Scripts\python phase3b_embeddings\word2vec.py` | `.venv\Scripts\python -m unittest discover -s phase3b_embeddings -v` |
| 4a | `.venv\Scripts\python phase4_transformer\attention_by_hand.py` and `.venv\Scripts\python phase4_transformer\inspect_model.py` | `.venv\Scripts\python -m unittest discover -s phase4_transformer -v` |
| 4b | `.venv\Scripts\python phase4_transformer\train.py` then `.venv\Scripts\python phase4_transformer\generate.py` | (same as 4a) |
| 5a | `.venv\Scripts\python phase5_generation\explore.py` | `.venv\Scripts\python -m unittest discover -s phase5_generation -v` |
| 5b | `.venv\Scripts\python phase5b_evaluation\run_eval.py` and `.venv\Scripts\python phase5b_evaluation\canary.py` | `.venv\Scripts\python -m unittest discover -s phase5b_evaluation -v` |

Guardrail design and the known-weaknesses record are in `phase5b_evaluation\README.md`.

| Phase | Run | Test |
|---|---|---|
| 6a | `.venv\Scripts\python phase6_finetuning\finetune.py` then `.venv\Scripts\python phase6_finetuning\preference.py` | `.venv\Scripts\python -m unittest discover -s phase6_finetuning -v` |

| 6b | `.venv\Scripts\python phase6b_tools\demo.py` (the model is already trained; `tool_model.py` retrains it in about 7.5 minutes) | `.venv\Scripts\python -m unittest discover -s phase6b_tools -v` |

| App | `.venv\Scripts\python app\server.py`, then open http://127.0.0.1:8765 | `.venv\Scripts\python -m unittest discover -s app -v` |

**Try it in the browser first:** the app joins every phase into one pipeline and shows each step for each request.
Start with `docs\FINAL_SUMMARY.md` and `docs\ARCHITECTURE.md` for the overall picture.

Phase 6a saves `phase6_finetuning\checkpoints\finetuned.pt` (instruction-tuned) and `preference.pt` (after DPO).
Both have exactly the same shape and tokenizer as the Phase 4b model; only the numbers inside changed.

The trained model is saved to `phase4_transformer\checkpoints\tinygpt.pt` (about 160 KB). It holds the model's
settings, its 35,469 learned numbers, the tokenizer, and the training history.

## Files

- `README.md`: this file
- `PROGRESS.md`: progress log (phases done, experiments run, known issues, next steps)
- `GLOSSARY.md`: terms, added as we learn them
- `check_env.py`: reports what's installed
- `requirements.txt`: the external libraries we'll install later
- `data/stories.txt`: 30 short stories we wrote, used for training from Phase 3 (see `data/README.md`)
- `data/relations.txt`: 98 sentences we wrote for training the word embeddings
- `JOURNEY.md`: the hurdles, and how they were solved
- `LIMITATIONS.md`: what the system can't do
- `docs/video-notes.md`: where the video simplifies or is inaccurate
