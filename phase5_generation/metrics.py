"""Phase 5a: simple, automatic measures of generated text.

None of these measures "quality" or "truth". Each catches one specific failure:
  real_word_rate   - is it inventing non-words? ("farmerily")
  repetition_rate  - is it stuck in a loop? ("fouse fouse fouse")
  longest_copy     - is it reciting training text word for word?
  distinct_outputs - does it say the same thing every time?
"""
import re

from sampling import ROOT  # noqa: F401  (sets up the import path)
from generate import longest_copied_run


def words(text):
    return re.findall(r"[a-z']+", text.lower())


def real_word_rate(text, known_words):
    """Fraction of generated words that appear somewhere in the training text."""
    ws = words(text)
    return sum(w in known_words for w in ws) / len(ws) if ws else 0.0


def repetition_rate(text, n=3):
    """Fraction of word n-grams (runs of n words) that already appeared earlier in the same text."""
    ws = words(text)
    grams = [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]
    if not grams:
        return 0.0
    seen, repeats = set(), 0
    for g in grams:
        repeats += g in seen
        seen.add(g)
    return repeats / len(grams)


def longest_copy(text, training_text):
    return len(longest_copied_run(text, training_text))


def distinct_outputs(texts):
    return len(set(texts))
