# How I built a small ChatGPT-style model from scratch

I'm **Dravid Albert**, a product manager. I wanted to know how AI models like ChatGPT really work, so I built a tiny one myself.
No ready-made AI model. No paid services. Just my laptop.

I decided what to build, how good it had to be and how to test it. Claude, an AI coding assistant, wrote the code with me.

## What I built

Think of it as a very small version of what sits behind a chatbot:

- **A model that learns to write** by reading stories
- **A model that can ask for help,** like asking a calculator to do a sum
- **Safety checks** that stop bad requests and bad answers
- **A set of tests** that try to break it
- **A web page** where you can type something and see every step behind the answer

The models are tiny, so the writing is often strange. The point was never to beat ChatGPT. It was to understand how these systems work and where they fail.

## How it works, in plain words

Here's what happens when you type something, and what I saw when I built each part:

| The idea | What it means | What I saw |
|---|---|---|
| **Tokenization** | The AI doesn't read words. It cuts text into small pieces called tokens, and each piece gets a number. AI services usually charge by the token | Two sentences of the same length took 9 tokens and 23 tokens. Words the model has seen often take fewer tokens, so unfamiliar text costs more |
| **Embeddings** | Each token gets a list of numbers, like a location on a map. Tokens used in similar ways end up close together | "King" and "queen" ended up close. So did "happy" and "sad", because the AI learns how words are used, not what they mean |
| **Attention** | Each token looks back at the earlier ones to work out what matters. It's how "bank" can lean towards "river" or "money" | In my hand-worked example, "bank" leaned towards nature after "river" and towards money after "money" |
| **Training** | The AI guesses the next token, checks how wrong it was and adjusts its numbers. Then it repeats this thousands of times | It went from random letters to real words and story openings in about 20 seconds |
| **Predicting the next token** | The AI gives every token a probability, picks one, adds it to the text and repeats. It has no plan for the whole sentence | After "Once upon a time", it gave " a" 57% and " there" 40% |
| **Temperature** | A creativity dial. Low plays it safe and repeats itself. High takes more chances and makes more mistakes | Real words fell from 82% at the lowest setting to 53% at a high one |
| **Context window** | How much text the AI can see at once. Anything older is cut off, as if it never existed | My story model sees only 32 tokens. When I swapped the cut-off text for a different story, the output didn't change at all |
| **System prompt** | Instructions the AI gets before your message. They only work if it was trained to follow instructions | Mine weren't trained for that, so these models ignored the rule |
| **Fine-tuning** | Extra training on examples, to teach a style or format | It learned the format, but it still made up the content |
| **Tool calling** | The AI writes a request, ordinary code does the work, and the AI uses the result | It asked for 40 + 2, and the code worked out 42 |
| **Guardrails** | Ordinary code that checks what goes in and what comes out | They block obvious problems but are easy to fool by rewording |

The context window also limited what I could build. "Tell me about the king." used 12 of the model's 32 tokens, so I had to change the format to "About the king."

## The decisions I made

**1. I said testing can't be an afterthought.**
From day one, I asked for tests with normal requests, strange requests and attempts to trick the system, and I wanted failures written down.
A nice demo would have hidden the biggest problems we found.

**2. I didn't trust a good score on its own.**
I asked for the model to be tested on stories it had never seen. That caught something big.
One version got a great score and wrote perfect sentences. But on new stories, it did worse than random guessing.
It hadn't learned to write. It had memorised a story and was repeating 48 words of it.
If we'd gone by that score, the worst version would have looked like the best.

**3. I didn't take shortcuts.**
I could have downloaded an existing AI model. I didn't, because I wanted to be able to explain every part. The core parts are our own code.

**4. I said important rules can't just be instructions to the AI.**
So we tested it. We told the model "never write about the king", then "always write about the king". Out of 100 tries, it mentioned the king 10 and 11 times. No real difference.
Then we added a simple code check, and no story mentioning the king got through.
If a rule must always hold, ordinary code has to enforce it.

**5. I made sure the AI can only ask, not act.**
The AI can't do anything by itself. It can only write a request, and our code decides whether to carry it out.
We tested this with a fake AI that got tricked into trying to delete something. Our code refused.
We also learned that if you approve everything automatically, that protection is gone. So the page shows you the exact action and asks you first.

**6. I asked to see the finished product.**
I wanted to use it on a real page, and I asked if it was really complete. That exposed a gap: the safety checks and the part that carries out actions had never been run together. We joined them, and that's what the page runs today.

**7. I asked the product question: what would actually make it better?**
The answer depends on the problem, and we had the test results to back it up:

| The problem | What fixes it | What we saw |
|---|---|---|
| Writing is poor | More reading material | Going from 27 stories down to 5 made it clearly worse. A bigger model on the same stories didn't help |
| Wrong facts | Let it look things up | "About the moon" went from a made-up answer to "I do not know." |
| Wrong format | Extra training on examples | Format went from 0 out of 6 to 6 out of 6, but the answers were still about the wrong thing |
| Rules that must always hold | Ordinary code | The code check worked every time. The instruction didn't |

**8. I kept asking how the pieces fit.**
Asking how RAG (letting an AI search documents before answering) fits here showed that our lookup only finds exact names. Ask about "the ocean" and it won't find "the sea". That's now on the list below.

## Problems we hit along the way

| What went wrong | How we fixed it |
|---|---|
| A test score hit 100%, but the model was writing "The rapy." | We read real answers now, not just scores |
| An instruction seemed to work in a small test of 20 | We ran 100 tries with the opposite instruction to compare, and the effect disappeared |
| Two of our tests were passing for the wrong reason | We rewrote them so they check what they claim to check |
| Answers that should have been identical kept changing | We found a hidden setting left switched on, fixed it and added tests so it can't come back |
| The AI misspelled words when asking for a lookup ("fok" for "fox") | We gave it more varied practice words. Correct requests went from 24 out of 36 to 36 out of 36 |
| The AI looked up the right fact, then garbled it | The page now spots this, warns you and shows the correct fact |
| The AI could memorise a secret it saw repeatedly | A fake secret repeated 5 times leaked every time in the memorising version, but never when we stopped training earlier. The real fix is removing secrets before training |
| The install failed on Windows, and the web page got stuck | We moved the project to a shorter folder path, and fixed how the page handles connections |

## The results

| What we checked | Result |
|---|---|
| Automatic tests | All 170 pass |
| Try-to-break-it test cases | 22 of 29 pass, and the other 7 are known weaknesses we've written down |
| Choosing the right action on new requests | 36 out of 36 |
| The AI's own final answer on those requests | 31 out of 36 |
| Fake bad actions we tried | 13 tried, and none changed any data |

These are small tests on tiny models, so they don't prove it's safe for real use. The full list of what it can't do is in [LIMITATIONS.md](LIMITATIONS.md).

## What I'd improve next

These aren't done yet. I'd do them in this order:

1. **Give it far more to read.** Train it on a large public collection of children's stories, with a slightly bigger model. This should help most with the nonsense and made-up words. It could take from tens of minutes to a few hours on my laptop. I'll compare old and new side by side, and only switch if the new one is better.
2. **Teach it to follow simple instructions, then add a system prompt.** A system prompt is the set of instructions an AI gets before your message. Today my models ignore those, because they were never taught to follow them. Once it's trained on examples, simple rules like "answer in one short sentence" could work, though not reliably.
3. **Let it search by meaning.** So asking about "the ocean" can find "the sea". This is a basic version of RAG.
4. **Teach it more ways of asking.** So "what is 40 plus 2" works, not just "add 40 and 2".
5. **Redo the question-and-answer training on the improved model.** It might finally answer about the right thing, which it never did in the first test.
