# Architecture: from request to response

This describes what we built. Production systems follow the same outline, with far bigger models,
more layers of checks, and details that aren't public.

## How a model gets made (ahead of time)

```
 stories.txt ──► BPE tokenizer training ──► tokenizer (vocabulary + merges)
                                                  │
 stories.txt ──► PRETRAINING: next-token prediction ──► base model (continues text)
                 validation stories pick the best step        │
                                                              ▼
 instructions.txt ──► FINE-TUNING: same model, graded ──► fine-tuned model (right format)
                      only on the answers                     │
                                                              ▼
 preferences.txt ──► PREFERENCE TUNING (DPO): chosen ──► preference-tuned model
                     up, rejected down, stay close
```

## What happens on each request (at run time)

```
 User types text
      │
      ▼
┌─ APPLICATION ─────────────────────────────────────────────────────────────────┐
│ INPUT CHECKS   type, empty, length, control characters, blocked words,        │
│                override patterns, unknown characters ──► refuse, or continue  │
│      │                                                                        │
│      ▼                                                                        │
│ BUILD CONTEXT  request + earlier lines; crop to the window                    │
│      │                                                                        │
│      ▼                                                                        │
│ ┌─ MODEL (weights file; can only output text) ──────────────────────────────┐ │
│ │ text ► tokens ► IDs ► token vector + position vector                      │ │
│ │   ► block × N [ attention (causal mask) + feed-forward ] ► logits         │ │
│ │   ► temperature ► softmax ► top-p ► pick one token ► append ► repeat      │ │
│ └───────────────────────────────────┬───────────────────────────────────────┘ │
│      ┌──────────────────────────────┴────────────┐                            │
│      ▼ line starts with "{"                      ▼ plain text                 │
│ VALIDATE CALL  JSON, known tool,            OUTPUT CHECKS  blocked words,     │
│   exact fields, types, ranges               copied text, quality              │
│      │                                           │                            │
│      ▼                                           ▼                            │
│ PERMISSIONS  granted? person approved?      answer shown to the user          │
│      │                                                                        │
│      ▼                                                                        │
│ EXECUTE ── THE EXECUTION BOUNDARY: only this application code can act         │
│      │                                                                        │
│      └── "result ..." or "error ..." appended; back to BUILD CONTEXT (max 3)  │
│                                                                               │
│ LOG every decision                                                            │
└───────────────────────────────────────────────────────────────────────────────┘
      ▲
 EVALUATION runs from outside: test suite, canary test, reviewing the logs
```

`app/pipeline.py` runs this whole path, and `app/server.py` shows it on a web page. It also checks that a final
answer matches the tool result. The guardrail switch on the page never turns off tool validation or permissions.

## The pipeline in one paragraph

Text is split into tokens, and each token ID picks a learned vector, with a position vector added. Blocks of
attention let each token gather information from earlier tokens, and feed-forward layers process each token
on its own. The last position's vector becomes one score (logit) per vocabulary token. Temperature, softmax
and top-p turn those scores into probabilities, and one token is picked. It's appended and the whole process
repeats until the answer is finished. The model never acts: if its text proposes a tool call, the application
validates it, checks permission, runs it, and appends the result so the model can write its answer. It keeps
no memory between requests, only the text the application sends it.

## What's inside a saved model file

`phase4_transformer/checkpoints/tinygpt.pt` is about 160 KB and holds four things:

| Part | Contents |
|---|---|
| `config` | Settings: 141 tokens, 32-token window, 32 dims, 4 heads, 2 blocks |
| `weights` | **35,469 learned numbers**, listed below |
| `tokenizer` | The vocabulary and the merges. A model is useless without its matching tokenizer |
| `info` | Training step, losses, settings and the training text (the copy detector uses it) |

| Weights | Shape | Numbers | Job |
|---|---|---|---|
| `token_embedding` | 141 × 32 | 4,512 | A vector for each token |
| `position_embedding` | 32 × 32 | 1,024 | A vector for each position. This table is why the window is 32 |
| `make_qkv` (each block) | 96 × 32 | 3,072 | Makes query, key and value |
| `combine` (each block) | 32 × 32 + 32 | 1,056 | Merges the heads |
| `feed_forward` (each block) | 128 × 32 + 128, 32 × 128 + 32 | 8,352 | Per-token processing |
| layer norms (each block) | 2 × (32 + 32) | 128 | Keep the numbers in a stable range |
| `final_norm` | 32 + 32 | 64 | Same, before the output |
| `to_logits` | 141 × 32 + 141 | 4,653 | Turns a vector into a score per token |

Nothing readable is stored in the weights: no sentences, facts or rules. Knowledge exists only as these
numbers, which is why the model can't look anything up, and why memorised text can only be found by testing.
The file doesn't contain code either: `model.py` is needed to use the numbers.

## Which control belongs where

| Need | Use | Because |
|---|---|---|
| A format, style or tone | Fine-tuning | Changes behaviour by default; doesn't add reliable facts |
| Facts that change, or must be right | A tool or retrieval | Comes from a source you control |
| Instructions for one request | Prompting | Cheap, flexible, followed often but not always |
| A rule that must always hold | Application code | The only layer enforced every time |
| Limiting what can go wrong | Permissions | Work even when the model is fooled |
| Knowing whether any of it works | Evaluation and monitoring | Finds problems; doesn't prevent them |
