# Limitations

What this system can't do, with the numbers we measured, and what would address each limit.
Measured on a Windows laptop (AMD Ryzen AI 7 350, 23 GB RAM, **CPU only**).

**This is a learning project. It isn't a useful chatbot, and it doesn't establish production-level safety.**
Results come from tiny models, tiny test sets and usually one run, so treat exact numbers as indicative.

## 1. The models are extremely small

| Model | Parameters | Training data | Training time |
|---|---|---|---|
| Story model | 35,469 | 27 stories we wrote, about 3,000 tokens | about 20 seconds |
| Fine-tuned model | 35,469 | 30 instruction pairs | seconds |
| Tool-calling model | 168,653 | generated conversations | about 7.5 minutes |

Real LLMs have billions of parameters or more. **Lessons about architecture carry over; output quality doesn't.**

**What would help:** far more training text first, then a bigger model. On this little data, a bigger model
did slightly worse, and a large public dataset hasn't been tried.

## 2. Output quality is poor

- The story model invents words: "a litttle girl lived in the farm. The farmerily."
- About 67% of generated words are real at the balanced setting (T=0.8, top-p 0.9), and "real" doesn't mean "right".
- Text is locally plausible but has no plot or logic.

## 3. The models don't know facts or understand anything

- The fine-tuned model has the right format 6/6 and is on topic 0/6. "About the moon." gets a made-up answer.
- Fine-tuning taught format. It doesn't provide facts, and it made story-writing worse (loss 3.21 to 3.36).
- Preference tuning reached 100% on its metric, but "happy" appeared in only 1% of answers, some reading "The rapy."
  The preferences were written by us, not gathered from human raters.

## 4. Prompts don't control these models

"Never write about the king" gave 10/100 mentions, "always" gave 11/100 and no rule gave 13/100. These models
weren't trained to follow instructions. There is no system prompt or conversation memory in the app.

## 5. Guardrails are simple and fail both ways

Word lists and text patterns only. Full record: `phase5b_evaluation/README.md`.

- **Missed:** rewording ("Forget what you were told"), disguises ("h-u-r-t") and synonyms ("harm").
- **Over-blocked:** "The knight did not hurt anyone".
- **Can't detect wrong answers:** "The dog of the prince was called" gets an invented answer with no warning.
- **Short secrets get through:** a fake secret leaked 21/21 from an overfitted model, under the 12-word copy limit.
- Thresholds are unvalidated guesses, and the 29 cases run with one seed each. The suite tests the Phase 5b
  guarded model and doesn't cover the app's full pipeline.

**What would help:** trained classifiers, labelled data for measuring error rates, and removing sensitive data before training.

## 6. Tool calling is a narrow demonstration

- Right call 36/36, right model answer 31/36, on one small test set from one training run.
- It only knows set phrasings: `add X and Y`, `about the X`, `delete the X`.
- Tools work on an in-memory dictionary, and deletions are lost when the server restarts.
- **A valid call can have the wrong argument**, and validation can't catch that. Approval has to show the exact call.
- **Blanket approval removes the protection.** A fake model obeying an injected instruction deleted data when delete was always approved.
- The answer check is a simple text match, and the attacks only cover what we thought of.

Full record: `phase6b_tools/README.md`.

## 7. No retrieval (RAG), and the search is exact-match

The lookup finds exact names only, so "ocean" won't find "sea". The Phase 3b word embeddings aren't connected to the
app. Each model's own token embeddings are used inside the model, not for search.

## 8. Fixed context windows

Input beyond the window has no effect at all. The story models see 32 tokens, and "Tell me about the king." alone took
12 of them with this vocabulary. The tool model sees 144 characters.

## 9. Language and character support

The story tokenizer knows English letters, a few capitals, comma, full stop and newline only. Anything else becomes
`<unk>`, or is blocked by the guardrails. Word embeddings capture usage, not meaning: "happy" and "sad" come out close.

## 10. The web app is for local use only

One request runs at a time, with no accounts, encryption, rate limiting or stored logs. It only listens on this computer,
and shouldn't be exposed to a network.

## 11. Not built or not tested

- Training on a large public dataset, and real RAG.
- Real human feedback, or classic RLHF with a reward model.
- A third, untouched test set. The best step is chosen on validation data, so those scores are slightly optimistic.
- GPU training, and other operating systems.

## 12. Source material and licences

- The project follows a YouTube video's structure, and its original code was not used. The transcript is **not included** in this repository.
- The training text was written for this project. Software dependencies keep their own licences.
- This repository doesn't have a licence file yet, so reuse rights aren't granted by default.
