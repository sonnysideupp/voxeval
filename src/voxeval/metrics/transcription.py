"""Transcription metrics — did the agent *hear* correctly?

Word error rate is the standard measure and it is not sufficient on its own. A
voice agent taking an order can score 6% WER and still be broken, because the
one word it lost was the product name. Conversely it can score 30% WER on filler
and hesitation while getting every consequential term right.

So this module pairs WER with **term recall**: did the words that carry the
transaction survive? Those are the dish names, street names, drug names, part
numbers — whatever your domain's nouns are.

Motivating incident: a caller said "Kung Pao Chicken" and the recognizer returned
"compound chicken". WER on that utterance was low. The order silently dropped the
item.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, NFKC-normalize.

    Applied to both reference and hypothesis before comparison so that "Kung-Pao"
    and "kung pao" are not counted as an error. Unicode normalization matters more
    than it looks: a curly apostrophe in "General Tso's" arriving as a straight one
    is not a transcription error, and counting it as one hides real ones.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, divided by reference length.

    Returns 0.0 for two empty strings, and 1.0 when the reference is empty but the
    hypothesis is not (everything present is an insertion). Values above 1.0 are
    possible when the hypothesis is much longer than the reference — that is
    correct behaviour, not a bug, and it is why callers should not assume WER is a
    percentage bounded at 100.
    """
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    # Two-row dynamic programming: O(len(ref) * len(hyp)) time, O(len(hyp)) space.
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            current[j] = min(
                previous[j] + 1,            # deletion
                current[j - 1] + 1,         # insertion
                previous[j - 1] + (r != h),  # substitution
            )
        previous = current
    return previous[len(hyp)] / len(ref)


@dataclass(frozen=True)
class Term:
    """A term that must survive transcription, with its accepted spellings.

    `variants` exists because the same item legitimately arrives several ways: a
    recognizer may return "kung pao", "kung-pao", "gong bao", or the Chinese
    characters, and all of them mean the caller was understood. Listing them is
    not grading leniently — it is refusing to fail on orthography.

    `collapses` is the opposite: known-wrong forms that indicate a specific
    mis-hear. Detecting these is far more actionable than a WER delta, because
    "compound chicken" tells you *which* failure happened.
    """

    name: str
    variants: tuple[str, ...] = ()
    collapses: tuple[str, ...] = ()

    def accepted(self) -> tuple[str, ...]:
        return (self.name,) + self.variants


@dataclass
class TermRecallResult:
    recall: float
    found: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    collapsed: list[str] = field(default_factory=list)

    @property
    def has_collapse(self) -> bool:
        """True if a known mis-hear appeared. Treat as a hard failure: it means the
        recognizer produced a plausible wrong answer rather than an obvious one."""
        return bool(self.collapsed)


def term_recall(hypothesis: str, terms: list[Term]) -> TermRecallResult:
    """Fraction of required terms that survived, plus any known collapses.

    Matching is substring-based on the normalized text rather than token-exact,
    because a multi-word term ("beef with broccoli") should be found inside a
    longer utterance without requiring the caller to have said nothing else.

    Non-Latin variants are also checked against the raw hypothesis, since
    normalization strips nothing from CJK text but tokenization is meaningless for
    it — "牛肉面" is one token to a human and several to `.split()`.
    """
    normalized = normalize(hypothesis)
    raw = hypothesis or ""

    found: list[str] = []
    missed: list[str] = []
    for term in terms:
        hit = any(
            normalize(v) in normalized or (not v.isascii() and v in raw)
            for v in term.accepted()
        )
        (found if hit else missed).append(term.name)

    collapsed = [
        c
        for term in terms
        for c in term.collapses
        if normalize(c) in normalized or (not c.isascii() and c in raw)
    ]

    total = len(terms)
    return TermRecallResult(
        recall=(len(found) / total) if total else 1.0,
        found=found,
        missed=missed,
        collapsed=collapsed,
    )


_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def extract_digits(text: str) -> str:
    """Digits from a spoken string, spelled-out numbers included.

    Phone numbers, order numbers, unit numbers and card digits are where a voice
    agent's errors become expensive and where WER is least informative: "five one
    seven" and "5 1 7" have a WER of 1.0 against each other and are the same
    number. Compare these with `digit_accuracy` instead.
    """
    out: list[str] = []
    for token in tokenize(text):
        if token in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[token])
        else:
            out.extend(c for c in token if c.isdigit())
    return "".join(out)


def digit_accuracy(reference: str, hypothesis: str) -> float:
    """1.0 only when every digit matches in order; otherwise 1 - (edit distance / len).

    Deliberately strict. A phone number that is 90% right is 100% wrong — nobody
    is called back.
    """
    ref, hyp = extract_digits(reference), extract_digits(hypothesis)
    if not ref:
        return 1.0 if not hyp else 0.0
    if ref == hyp:
        return 1.0

    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return max(0.0, 1.0 - previous[len(hyp)] / len(ref))
