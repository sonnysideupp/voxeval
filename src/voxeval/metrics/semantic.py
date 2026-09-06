"""Semantic scoring — did the meaning survive, even if the words changed?

Word error rate compares strings. Callers and recognizers do not agree on
strings, and most of the disagreement is meaningless:

    reference   "I'd like two coffees at six thirty"
    hypothesis  "I would like 2 coffees at 6:30"

That is a *perfect* transcription and a WER of roughly 57%. Every difference is
orthographic — a contraction, a numeral, a colon. Score enough utterances like
this and the suite-wide WER sits near the threshold permanently, so a real
regression has nowhere to move and the gate stops meaning anything.

Two layers here, in order of how much you should trust them:

**1. Deterministic normalization (`semantic_wer`).** Expand contractions, write
numbers as digits, drop disfluencies, then run the ordinary WER. Free, instant,
reproducible, and it removes the bulk of the fake errors. This is the one to gate
on.

**2. A similarity callable (`Similarity`).** For genuine paraphrase — "I'd like
to order" versus "can I get" — no normalization will help, because the words
really are different and the meaning really is the same. That needs a model.
It is optional, it costs money, and it is non-deterministic, so it defaults to a
WARN and ships with `token_f1` as a zero-dependency floor.

The order matters. Reach for the model only after normalization, or you will pay
an API call to discover that "6:30" and "six thirty" are the same time.
"""

from __future__ import annotations

import unicodedata
from typing import Protocol, runtime_checkable

from .transcription import tokenize, word_error_rate

# Expanded rather than contracted, so "I'd"/"I would" converge. This is an
# explicit whitelist and must stay one: the tempting generic rule ("'s" -> " is")
# turns "General Tso's Chicken" into "General Tso is Chicken" and invents an
# error in exactly the domain terms this library exists to protect.
_CONTRACTIONS: dict[str, str] = {
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "you'd": "you would", "you'll": "you will", "you're": "you are",
    "you've": "you have", "we'd": "we would", "we'll": "we will",
    "we're": "we are", "we've": "we have", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have",
    "he'd": "he would", "he'll": "he will", "he's": "he is",
    "she'd": "she would", "she'll": "she will", "she's": "she is",
    "it'd": "it would", "it'll": "it will", "it's": "it is",
    "that's": "that is", "there's": "there is", "here's": "here is",
    "what's": "what is", "who's": "who is", "let's": "let us",
    "can't": "cannot", "won't": "will not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "wouldn't": "would not", "couldn't": "could not",
    "shouldn't": "should not", "mustn't": "must not", "ain't": "is not",
    # Casual speech forms. Recognizers differ on whether they expand these, and
    # the difference is never the agent's fault.
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "lemme": "let me", "gimme": "give me", "kinda": "kind of",
    "sorta": "sort of", "cuz": "because", "coz": "because", "'cause": "because",
    "ok": "okay", "yeah": "yes", "yep": "yes", "yup": "yes", "nope": "no",
}

# Only unambiguous disfluencies. "like" and "you know" are deliberately absent:
# "I'd like a coffee" is not a filler, and stripping it would delete the verb
# that carries the request.
_DISFLUENCIES: frozenset[str] = frozenset({
    "um", "umm", "uhm", "uh", "uhh", "er", "err", "ah", "ahh",
    "mm", "mmm", "hmm", "hm", "mhm", "eh",
})

_UNITS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _strip_apostrophes(text: str) -> str:
    """Fold every apostrophe variant to ASCII so the contraction map hits.

    A curly apostrophe is what a TTS engine emits and a straight one is what a
    recognizer returns. Without this fold, "I’d" misses the map, survives to
    punctuation-stripping as "i d", and scores as two substitutions.
    """
    return text.replace("’", "'").replace("ʼ", "'").replace("´", "'")


def _spoken_numbers_to_digits(tokens: list[str]) -> list[str]:
    """Fold spoken number words into digit strings.

    Handles the compounds that actually occur in orders and times: "twenty five"
    -> "25", "six thirty" -> "6 30". Deliberately conservative — it does not
    attempt "one hundred and twenty three", because a wrong fold invents an error
    in a quantity, which is worse than leaving two tokens unmatched.

    "six thirty" stays two tokens rather than becoming "630", so it matches a
    hypothesis of "6:30" once punctuation is stripped.
    """
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _TENS:
            # "twenty five" -> 25, but "thirty" alone -> 30.
            if i + 1 < len(tokens) and tokens[i + 1] in _UNITS and 1 <= _UNITS[tokens[i + 1]] <= 9:
                out.append(str(_TENS[token] + _UNITS[tokens[i + 1]]))
                i += 2
                continue
            out.append(str(_TENS[token]))
        elif token in _UNITS:
            out.append(str(_UNITS[token]))
        else:
            out.append(token)
        i += 1
    return out


def semantic_normalize(
    text: str,
    expand_contractions: bool = True,
    fold_numbers: bool = True,
    drop_disfluencies: bool = True,
) -> str:
    """Reduce an utterance to the form where only meaningful differences remain.

    Order is load-bearing: apostrophes are folded and contractions expanded while
    punctuation still exists, because the ordinary `normalize()` deletes the
    apostrophe that identifies a contraction in the first place.
    """
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = _strip_apostrophes(text)

    if expand_contractions:
        # Split on whitespace only — punctuation is still present here by design,
        # so strip it per-token for the lookup and keep the expansion.
        rebuilt: list[str] = []
        for raw in text.split():
            # Quotes and brackets too, not just sentence punctuation: a token
            # arriving as “can't” misses _CONTRACTIONS and degrades to "can t".
            #
            # The straight apostrophe is deliberately NOT in this set. Stripping
            # it turns "'cause" into "cause", which is not in the map, so the
            # expansion to "because" is lost — a fix for quoting that breaks
            # elision. Curly quotes are stripped because they are almost always
            # quotation; the straight one is more often a contraction marker.
            stripped = raw.strip('.,!?;:"“”‘’()[]{}<>')
            rebuilt.append(_CONTRACTIONS.get(stripped, raw))
        text = " ".join(rebuilt)

    tokens = tokenize(text)  # now safe: lowercases, strips punctuation, collapses space

    if drop_disfluencies:
        tokens = [t for t in tokens if t not in _DISFLUENCIES]
    if fold_numbers:
        tokens = _spoken_numbers_to_digits(tokens)
    return " ".join(tokens)


def semantic_wer(reference: str, hypothesis: str) -> float:
    """Word error rate after semantic normalization.

    Always <= the raw WER on the same pair, because normalization only ever
    merges forms. Report both: raw WER tells you what the recognizer literally
    emitted, semantic WER tells you whether it mattered. A large gap between them
    is not a problem — it is the formatting noise you just stopped gating on.
    """
    return word_error_rate(semantic_normalize(reference), semantic_normalize(hypothesis))


@runtime_checkable
class Similarity(Protocol):
    """Scores meaning-equivalence of two utterances in [0.0, 1.0].

    A bare function satisfies it, so an embedding model is a lambda:

        def sim(ref, hyp):
            a, b = embed(ref), embed(hyp)
            return cosine(a, b)

    Use this only for genuine paraphrase. Anything normalization can fix should
    be fixed by normalization — it is free and it does not vary between runs.
    """

    def __call__(self, reference: str, hypothesis: str) -> float: ...


def token_f1(reference: str, hypothesis: str) -> float:
    """Bag-of-words F1 over semantically normalized tokens.

    The zero-dependency default for `Similarity`. It is a *floor*, not a
    substitute for an embedding model: it cannot see that "can I get" and "I'd
    like to order" mean the same thing, because they share no tokens.

    What it is genuinely good at is the failure embeddings are worst at —
    detecting that content words went missing. A truncated transcript that drops
    half the order scores badly here and can still embed at 0.9, because the
    surviving half is topically identical. Which is exactly why this stays in the
    library even once you wire up a real model.
    """
    ref = tokenize(semantic_normalize(reference))
    hyp = tokenize(semantic_normalize(hypothesis))
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0

    # Multiset intersection: saying "coffee" twice when one was ordered should
    # not count as two matches.
    remaining = list(hyp)
    overlap = 0
    for token in ref:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    if overlap == 0:
        return 0.0

    precision = overlap / len(hyp)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def meaning_preserved(reference: str, hypothesis: str, threshold: float = 0.85) -> bool:
    """Convenience check: did this utterance survive in substance?

    Uses `token_f1` so it stays deterministic and free. Pass a real `Similarity`
    to the suite instead when you need paraphrase tolerance.
    """
    return token_f1(reference, hypothesis) >= threshold
