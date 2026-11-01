# Data

| File | What it is | Source |
|---|---|---|
| `stories.txt` | 30 short children's stories, about 5 sentences each, in plain simple English | **Written by us for this project** (inspired by the video's description of its "30 short children's stories"; not copied from it) |
| `relations.txt` | 98 one-line sentences about royalty, people, cats and dogs, food, the kitchen and feelings | **Written by us for this project.** Modelled on the video's description of its embedding corpus (pairs like "king is a man who rules" / "queen is a woman who rules") |
| `instructions.txt` | 36 instruction → answer pairs ("About the owl." / "The owl was wise and old.") | **Written by us**, with facts taken from `stories.txt`. Used for fine-tuning (Phase 6a). Uses only characters the Phase 4b tokenizer knows (there's no "?" or ":") |
| `preferences.txt` | 26 preference pairs: an instruction, a **chosen** answer ("...was happy.") and a **rejected** one ("...was sad.") | **Written by us**, standing in for human rankings. A real preference dataset comes from many human raters. Used in Phase 6a |

`relations.txt` is built so that matching words appear in matching sentences, with only one word
changed (he/she, king/queen, cat/dog). That's what lets a tiny embedding model learn relationships.
Real corpora don't line up this neatly, which is one reason real embedding models need billions of words.

The stories deliberately reuse a small cast (king, queen, prince, princess, cat, dog, forest, castle)
so that a tiny model can find patterns. That also means anything trained on this file knows nothing
outside that small world.
