"""Pure lexical primitives, shared by every metric that compares words.

These sit below the metrics rather than inside any one of them, because both
transcription scoring and semantic normalization need the same tokenizer and the
same morphology. Putting `fold_plural` next to WER and the contraction map next
to term matching is how you end up with two English-morphology layers that drift
apart — which is exactly what happened before this module existed.

Nothing here knows what a metric is. No dependencies beyond the stdlib.
"""

from __future__ import annotations

import unicodedata


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


# Plurals that no suffix rule reaches. Listed plural -> singular so folding is
# symmetric: "leaves" maps to "leaf", and "leaf" is already its own key form, so
# both sides converge without the rule needing to know which way it was written.
#
# Scoped to words that plausibly appear in the product nouns this library
# guards — food, quantities, people. A general English pluralizer is a
# dependency and a maintenance surface; this is a lookup table.
_IRREGULAR_PLURALS: dict[str, str] = {
    "children": "child", "people": "person", "men": "man", "women": "woman",
    "feet": "foot", "teeth": "tooth", "mice": "mouse", "geese": "goose",
    "leaves": "leaf", "loaves": "loaf", "halves": "half", "knives": "knife",
    "wives": "wife", "shelves": "shelf", "lives": "life", "calves": "calf",
    "potatoes": "potato", "tomatoes": "tomato", "heroes": "hero",
    "fish": "fish", "sheep": "sheep", "series": "series", "species": "species",
}


def fold_plural(token: str) -> str:
    """Reduce an English plural to a canonical singular form.

    Morphological normalization lives here rather than in the matching code,
    because it is the same *kind* of operation as expanding a contraction or
    folding a spoken number: a difference in surface form that carries no
    difference in meaning.

    Applied to both sides of a comparison, so it never has to be *correct*
    English — only consistent. "glass" keeps its double-s and "hamster" has no
    trailing s, which is what stops this from turning into prefix matching where
    Term("ham") would match "hamster".
    """
    if token in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[token]
    # "-es" after a sibilant: glasses, boxes, dishes, sandwiches. Checked first,
    # because "glass" ends in "ss" and the plain "-s" rule below leaves it alone.
    if len(token) > 3 and token.endswith("es"):
        stem = token[:-2]
        if stem.endswith(("ss", "x", "z", "ch", "sh")):
            return stem
    if len(token) > 2 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    # Final canonicalization of the y/ie endings, applied to BOTH spellings so
    # they converge without the fold having to know which it started from.
    #
    # This is the case a plain "-ies -> -y" rule gets wrong: "berries" is berry
    # + ies, but "cookies" is cookie + s, and nothing in the plural distinguishes
    # them without a dictionary. Folding both endings to "i" sidesteps the
    # question entirely — berry/berries both land on "berri", cookie/cookies both
    # on "cooki", fry/fries both on "fri". None are words; all are consistent,
    # which is the only property this needs.
    if token.endswith("ie"):
        return token[:-2] + "i"
    if token.endswith("y"):
        return token[:-1] + "i"
    return token


