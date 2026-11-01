# Phase 6b: Tool calling

The model can only write text. One line of that text may *propose* an action. The application decides
whether to run it. The tools act on an in-memory dictionary only: no files, network or real data.

## Run

```
.venv\Scripts\python phase6b_tools\tool_model.py
.venv\Scripts\python phase6b_tools\demo.py
.venv\Scripts\python phase6b_tools\demo.py --request "add 40 and 2"
.venv\Scripts\python -m unittest discover -s phase6b_tools -v
```

Training takes about 7.5 minutes on CPU, and a trained model is already saved. The tests don't need it.

## Results (168,653 parameters, 3,000 steps, seed 0)

Requests never used in training, through the real harness with read-only permissions:

| Kind | Requests | Right call | Right final answer |
|---|---|---|---|
| add | 20 | 20 | 20 |
| about | 10 | 10 | 6 |
| delete | 6 | 6 | 5 |
| **Total** | **36** | **36** | **31** |

All 13 scripted bad proposals and the endless-call loop were stopped without changing any data.
The 36 requests are a small test set, and the numbers come from one training run.

## Known weaknesses

| # | Weakness | Evidence | Possible mitigation |
|---|---|---|---|
| T1 | The harness alone doesn't check the final answer against the tool result | "about the fox": correct lookup, then "The fox dand shed in f." | **Fixed in `app/pipeline.py`:** a mismatch is flagged and the tool's result is shown. The check is a simple text match |
| T2 | A valid call can have the wrong argument | an earlier training run proposed "fok" for "fox", which passes validation | Show the exact call for approval; compare arguments with the request; make actions reversible |
| T3 | Blanket approval removes the protection | gullible model + always-approve deleted the fox | Approve individual actions, and watch for approval fatigue |
| T4 | Injected text can reach the user even when no tool runs | tool results are data, and the model can repeat them | Treat tool output as untrusted when displaying it, too |
| T5 | Scripted attacks only test what we thought of | 13 cases | Keep adding cases, review logs, and get independent testing |
| T6 | The model was trained on one fixed phrasing per tool | "add 40 and 2" only | Varied training data |
| T7 | This folder alone isn't combined with the Phase 5b guardrails | separate code paths | **Fixed in `app/pipeline.py`:** one shared pipeline |

## Lessons from building it

- The Phase 6a model couldn't be reused. Its tokenizer has no digits, braces or quotes, and its window is too small.
- A first run used only pronounceable made-up words, and the model mistyped real ones ("fok", "libaray").
  More varied training words fixed those: right calls went from 24/36 to 36/36.
- That change made some conversations too long for the 128-character window. The model rejected them and
  training crashed. The window is now 144.
- We keep the last training step, because picking the best step using these test requests would flatter the score.
