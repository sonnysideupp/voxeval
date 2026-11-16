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

    Combining marks are kept. NFKC already composes the accents Latin scripts have
    precomposed forms for; the marks left are spelling, not decoration — the
    Devanagari vowel signs that make दिन (day) and दीन (poor) different words.
    `isalnum()` rejects them, so dropping non-alphanumerics alone erased them.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(c.lower() if _kept(c) else " " for c in text)
    return " ".join(text.split())


def _kept(c: str) -> bool:
    return c.isalnum() or c.isspace() or unicodedata.category(c).startswith("M")


# Scripts written without spaces between words, where token boundaries don't exist
# and a term can only be found as a substring.
_UNSEGMENTED_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "THAI", "LAO", "KHMER", "MYANMAR", "TIBETAN")


def _unsegmented(text: str) -> bool:
    return any(unicodedata.name(c, "").startswith(_UNSEGMENTED_SCRIPTS) for c in text)


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

    @property
    def measured(self) -> bool:
        """False when no term was required, so there was nothing to recall.

        A "Yes." clip scored against a menu vocabulary has recall 1.0 by definition.
        Averaging that in would pad the suite's term recall with turns that never
        tested it, so aggregation skips unmeasured results."""
        return bool(self.found or self.missed)


def _singular(token: str) -> str:
    """Fold a trailing plural 's' so "spring roll" matches "spring rolls".

    Applied to both sides, so it never needs to be *correct* English — only
    consistent. "glass" keeps its double-s and "hamster" has no trailing s, which
    is what stops this from undoing the token-boundary fix above: prefix matching
    would make Term("ham") match "hamster" again.

    Irregular plurals are not handled — "child"/"children", "leaf"/"leaves". They
    are rare in the product nouns this metric guards, and the escape hatch is
    already there: declare the other form in `Term.variants`.
    """
    # "-es" after a sibilant: glasses, boxes, dishes, sandwiches. Checked first,
    # because "glass" ends in "ss" and the plain "-s" rule below leaves it alone.
    if len(token) > 3 and token.endswith("es"):
        stem = token[:-2]
        if stem.endswith(("ss", "x", "z", "ch", "sh")):
            return stem
    if len(token) > 2 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _matches(variant: str, hypothesis_tokens: list[str], raw: str) -> bool:
    """Is this variant present as a whole token sequence?

    Token-boundary matching, not substring. Plain `in` reports Term("ham") as
    found in "champagne" and Term("tea") in "steak" — a false positive on the one
    metric that exists to prove the transaction words survived. A false "found"
    is worse than a miss, because it reads as success.

    Multi-word terms still match inside longer utterances ("beef with broccoli"
    within a full sentence); they just have to align to token boundaries.

    Scripts written without spaces use substring matching instead: `.split()` is
    meaningless for them, so "牛肉面" is one word to a reader and several tokens to
    the tokenizer. The choice is made after normalizing, on what the variant *is*
    rather than whether its bytes are ASCII: "ｔｅａ" folds to "tea" and must keep
    token boundaries, and "CAFÉ", "Café" or a decomposed "café" must all match
    `Term("café")` — comparing raw text dropped spoken terms from scoring.
    """
    text = normalize(variant)
    if not text:
        return False
    if _unsegmented(text):
        return text in normalize(raw)
    needle = [_singular(t) for t in text.split()]
    if not needle:
        return False
    haystack = [_singular(t) for t in hypothesis_tokens]
    n = len(needle)
    return any(
        haystack[i:i + n] == needle
        for i in range(len(haystack) - n + 1)
    )


def term_recall(
    hypothesis: str, terms: list[Term], reference: str | None = None
) -> TermRecallResult:
    """Fraction of required terms that survived, plus any known collapses.

    With a `reference`, a term is required only if the speaker actually said it.
    That is what lets one domain vocabulary — a menu, a catalog — be passed to every
    case: a clip of "Yes." cannot lose "kung pao chicken", because nobody said it.
    Requiring every term regardless scored a perfectly transcribed "Yes." 0/2 in a
    live run against Deepgram and failed the deploy gate for a recognizer that
    missed nothing. Without a reference there is no way to know what was said, so
    every term listed is required: pass only the terms that utterance contains.

    A collapse the speaker actually said is not a mis-hear and is not reported.

    Matching is on whole token sequences — see `_matches` for why substring
    matching is the wrong tool here.
    """
    raw = hypothesis or ""
    hypothesis_tokens = tokenize(hypothesis)

    def spoken(variant: str) -> bool:
        return reference is not None and _matches(variant, reference_tokens, reference)

    reference_tokens = tokenize(reference) if reference is not None else []
    required = (
        terms
        if reference is None
        else [t for t in terms if any(spoken(v) for v in t.accepted())]
    )

    found: list[str] = []
    missed: list[str] = []
    for term in required:
        hit = any(_matches(v, hypothesis_tokens, raw) for v in term.accepted())
        (found if hit else missed).append(term.name)

    collapsed = [
        c
        for term in terms
        for c in term.collapses
        if _matches(c, hypothesis_tokens, raw) and not spoken(c)
    ]

    total = len(required)
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
