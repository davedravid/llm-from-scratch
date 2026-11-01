# The journey: hurdles and how they were solved

This project was built by **Dravid Albert** (product manager, project owner) with an AI coding
assistant (Claude) writing the code. The goal wasn't a useful chatbot. It was to understand how an LLM
is built by building a very small one. This page explains the problems we hit and how each was solved:
what the owner decided, and what was fixed in the code.

## The big turning points (decided by the project owner)

**1. "Teach me one phase at a time, and test what fails."**
The owner set the terms before any code existed: strict phases with approval between them, experiments
rather than lectures, corrections wherever the source video was wrong, and evaluation and safety as core
work. That brief is why the project records its failures and known weaknesses alongside its results.

**2. "Don't use a pretrained model as a substitute."**
The transformer, tokenizer, training loop, sampling, guardrails and tool harness are all our own code.
PyTorch only supplies low-level maths, so every behaviour here can be traced to code in this repository.

**3. "Show me the final output, and we'll work backward from there."**
After the phases were built, the owner chose to start again from what the system actually does, rather than
reviewing phase by phase. Seeing the results first made the concepts easier to connect to something real.

**4. "I want to try it in the frontend, and I want all the phases complete."**
This request exposed a real gap: the guardrails and the tool-calling code had been built and tested
separately, and had never run together. Joining them into one pipeline behind a web page closed that gap.
The page also shows every step of a request, which makes it a tool for learning and not just a demo.

**5. "As a PM, how do I make this better?"**
This turned the discussion from settings to levers. The experiments had already answered it: fine-tuning
teaches format, data drives quality, tools and retrieval provide facts, and application code enforces rules.

**6. "How does RAG fit here? Are we using embeddings?"**
These questions showed that the lookup tool has the core idea of RAG but matches exact names only, and that
the Phase 3b embeddings aren't connected to the app. That's now recorded as a limit, not left implied.

## Technical hurdles along the way, and how they were solved

**Setup on Windows**
- Creating the Python environment failed. The first working folder had a 180-character path, which pushed
  pip's files past Windows' 260-character limit. The project moved to a short path outside OneDrive.

**Training and overfitting (Phase 4b)**
- The first model memorised the stories: training loss 0.13, validation loss 10.6, worse than random guessing
  (6.17). It wrote flawless prose that was a 48-word copy of a training story. The fix: keep the best
  validation step, use a smaller vocabulary and add dropout.
- Validation losses were first compared across tokenizers, which is unfair because predicting a letter is
  easier than predicting a word. Adding loss per character corrected the comparison.
- The tokenizer had been built from all the stories, so it was rebuilt from the training stories only,
  keeping the validation stories out of the vocabulary.

**Tests that proved nothing, or the wrong thing (Phases 5a and 5b)**
- A context-window check compared text with itself, so it could never fail. It was replaced with a real test:
  swap the dropped text for a different story and confirm the output stays identical.
- A hallucination case "passed" only because its words were unfamiliar. It was rewritten using familiar words,
  which exposed the real weakness: the guardrails can't detect a confident wrong answer.
- An evaluation case failed because the expectation was wrong, not the code. It was reviewed and corrected.

**Measuring instead of guessing**
- With 20 samples, a prompt rule seemed to work (mentions of the king fell from 4 to 1). With 100 samples and
  an opposite-rule control, "never" gave 10, "always" gave 11 and no rule gave 13. The model wasn't obeying.
- Preference tuning reached 100% on its metric while the model wrote "The rapy." The metric only scored two
  fixed answers, so real outputs are now read alongside it. The 0% starting score turned out to be a length
  bias: " happy" is 5 tokens and " sad" is 2.

**Fine-tuning (Phase 6a)**
- The first prompt format didn't fit. "Tell me about the king." took 12 of the 32 tokens, so the format was
  shortened to "About the king." with one-sentence answers.
- A format check also required the right subject, which hid a key result. Separating format from content showed
  6/6 on format and 0/6 on topic.
- Evaluation helpers left models in training mode, which made "repeatable" generation random. It was spotted
  when a score changed between runs. The helpers now restore the mode, and regression tests guard against it.

**Tool calling (Phase 6b)**
- The fine-tuned model's tokenizer has no digits, braces or quotes, so it couldn't write a tool call at all.
  This phase uses a new model built from the same code.
- The first tool model mistyped real words ("fok" for "fox"), getting 24 of 36 calls right, because its made-up
  training words were all pronounceable. More varied words raised that to 36 of 36.
- Those longer words overflowed the 128-character window and crashed training, so the window became 144.
- The model sometimes garbled a correct tool result. The app now flags a mismatch and shows the tool's result.

**The web app**
- With guardrails switched off, an empty prompt crashed the model. It now gets a clear error.
- The page hung because browsers open spare connections and the single-threaded server waited on an idle one.
  Connections are now handled in parallel, with one request at a time using the models.

**Security choices**
- Model files load in PyTorch's safe mode, because unsafe loading of an untrusted file can run code. When saving
  a Python path object broke loading, the path was stored as text and the safety setting stayed on.
- Turning off the guardrails on the page never turns off tool validation or permissions.

## How the work was checked

- 170 automatic tests across 10 suites, including cases that record known weaknesses.
- An evaluation suite of 29 normal, edge-case and adversarial cases: 22 pass, 7 are documented weaknesses, 0 unexpected failures.
- A canary test that plants a fake secret in the training data and checks whether it leaks.
- The web page was tried in a browser, including at phone width.
