"""Semantic normalization and similarity."""

from __future__ import annotations

import pytest

from voxeval.metrics.semantic import (
    meaning_preserved,
    semantic_normalize,
    semantic_wer,
    token_f1,
)
from voxeval.metrics.transcription import word_error_rate


class TestSemanticNormalize:
    def test_expands_contractions(self):
        assert semantic_normalize("I'd like it") == "i would like it"

    def test_folds_curly_apostrophes(self):
        # A TTS engine emits U+2019; a recognizer returns U+0027. Without the
        # fold the contraction map misses and "i’d" survives as two tokens.
        assert semantic_normalize("I’d like it") == semantic_normalize("I'd like it")

    def test_possessive_is_not_treated_as_a_contraction(self):
        # The regression this guards: a generic "'s -> is" rule turns the dish
        # name into "general tso is chicken" and invents an error in exactly the
        # kind of term the library exists to protect.
        assert semantic_normalize("General Tso's Chicken") == "general tso s chicken"

    def test_spoken_numbers_become_digits(self):
        assert semantic_normalize("two coffees") == "2 coffees"

    def test_compound_tens(self):
        assert semantic_normalize("twenty five rolls") == "25 rolls"

    def test_bare_tens_survive_alone(self):
        assert semantic_normalize("thirty rolls") == "30 rolls"

    def test_time_stays_two_tokens_to_match_punctuated_form(self):
        # "6:30" loses its colon in tokenization, so the reference must not
        # collapse to "630" or the two would never match.
        assert semantic_normalize("six thirty") == semantic_normalize("6:30") == "6 30"

    def test_drops_disfluencies(self):
        assert semantic_normalize("um I uh want it") == "i want it"

    def test_keeps_like_as_a_verb(self):
        # "like" is a filler in some speech and the main verb in "I'd like a
        # coffee". Stripping it would delete the request.
        assert "like" in semantic_normalize("I would like a coffee")

    def test_empty_input(self):
        assert semantic_normalize("") == ""

    def test_flags_disable_each_layer(self):
        assert semantic_normalize("I'd like two", expand_contractions=False).startswith("i d")
        assert "two" in semantic_normalize("I'd like two", fold_numbers=False)
        assert "um" in semantic_normalize("um hello", drop_disfluencies=False)


class TestSemanticWer:
    def test_formatting_only_differences_score_zero(self):
        ref, hyp = "I'd like two coffees at six thirty", "I would like 2 coffees at 6:30"
        assert word_error_rate(ref, hyp) > 0.3   # raw WER punishes it
        assert semantic_wer(ref, hyp) == 0.0     # nothing meaningful differs

    def test_real_word_loss_still_scores(self):
        # Normalization must not launder an actual missing item.
        assert semantic_wer("two coffees and a muffin", "two coffees") > 0.0

    def test_never_exceeds_raw_wer(self):
        for ref, hyp in [
            ("I'd like two", "I would like 2"),
            ("hello there", "goodbye there"),
            ("um yes", "yes"),
            ("", ""),
        ]:
            assert semantic_wer(ref, hyp) <= word_error_rate(ref, hyp) + 1e-9

    def test_paraphrase_is_not_handled(self):
        # Documents the boundary: normalization is lexical. Genuine paraphrase
        # needs a model, which is what Similarity is for.
        assert semantic_wer("can I get a coffee", "I would like to order a coffee") > 0.5


class TestTokenF1:
    def test_identical(self):
        assert token_f1("two coffees", "2 coffees") == 1.0

    def test_disjoint(self):
        assert token_f1("alpha", "beta") == 0.0

    def test_both_empty(self):
        assert token_f1("", "") == 1.0

    def test_one_empty(self):
        assert token_f1("hello", "") == 0.0
        assert token_f1("", "hello") == 0.0

    def test_penalizes_dropped_content(self):
        # The failure embeddings are worst at: a truncated transcript stays
        # topically identical and embeds high, but loses half the order.
        full = "one kung pao chicken and two spring rolls and a diet coke"
        assert token_f1(full, "one kung pao chicken") < 0.7

    def test_repetition_does_not_inflate_overlap(self):
        # Multiset intersection: saying "coffee" twice when one was ordered is
        # not two matches.
        assert token_f1("coffee", "coffee coffee coffee") < 1.0

    def test_bounded(self):
        for ref, hyp in [("a b c", "a b"), ("x", "x y z"), ("a", "a")]:
            assert 0.0 <= token_f1(ref, hyp) <= 1.0


class TestMeaningPreserved:
    def test_formatting_variation_passes(self):
        assert meaning_preserved("I'd like two coffees", "I would like 2 coffees")

    def test_dropped_item_fails(self):
        assert not meaning_preserved("two coffees and a muffin", "two coffees")

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_threshold_is_honored(self, threshold):
        result = meaning_preserved("a b", "a b", threshold=threshold)
        assert result is (threshold <= 1.0)


class TestQuotedContractions:
    def test_curly_quoted_contraction_still_expands(self):
        assert semantic_normalize('“can’t” do it') == "cannot do it"

    def test_double_quoted_token(self):
        assert semantic_normalize('"two" coffees') == "2 coffees"

    def test_leading_apostrophe_elision_is_preserved(self):
        # The straight apostrophe is deliberately not stripped: doing so turns
        # "'cause" into "cause", losing the expansion to "because".
        assert semantic_normalize("'cause I said so") == "because i said so"

    def test_possessive_still_untouched(self):
        assert semantic_normalize("General Tso's") == "general tso s"


class TestFoldPlural:
    """English morphology lives in one place now. It used to be split between a
    private `_singular` in transcription.py and the contraction/number map here —
    two layers of the same kind of normalization, free to drift apart."""

    def test_regular_and_es_plurals(self):
        from voxeval.metrics import fold_plural
        for plural, singular in [("rolls", "roll"), ("glasses", "glass"),
                                 ("boxes", "box"), ("dishes", "dish"),
                                 ("sandwiches", "sandwich"), ("roses", "rose")]:
            assert fold_plural(plural) == singular

    def test_irregular_plurals(self):
        from voxeval.metrics import fold_plural
        for plural, singular in [("leaves", "leaf"), ("knives", "knife"),
                                 ("halves", "half"), ("loaves", "loaf"),
                                 ("children", "child"), ("people", "person"),
                                 ("potatoes", "potato"), ("tomatoes", "tomato")]:
            assert fold_plural(plural) == singular

    def test_unchanged_forms(self):
        from voxeval.metrics import fold_plural
        # Already singular, invariant, or simply not a plural. "hamster" is the
        # one that matters: folding must never become prefix matching.
        for word in ["ham", "hamster", "glass", "fish", "sheep", "series"]:
            assert fold_plural(word) == word

    def test_folding_is_idempotent(self):
        from voxeval.metrics import fold_plural
        for word in ["leaves", "glasses", "rolls", "children", "ham"]:
            assert fold_plural(fold_plural(word)) == fold_plural(word)

    def test_singular_and_plural_converge(self):
        # The property that makes symmetric folding safe: both spellings land on
        # the same key, so it never has to be correct English.
        from voxeval.metrics import fold_plural
        for a, b in [("leaf", "leaves"), ("knife", "knives"), ("roll", "rolls"),
                     ("glass", "glasses"), ("child", "children")]:
            assert fold_plural(a) == fold_plural(b)


class TestIesPlurals:
    """A plain "-ies -> -y" rule cannot work: "berries" is berry+ies but
    "cookies" is cookie+s, and the plural alone does not distinguish them.
    Folding both endings to "i" sidesteps it — none of the results are words,
    all of them are consistent, and consistency is the only property needed."""

    def test_y_plurals_converge(self):
        from voxeval.metrics import fold_plural
        for singular, plural in [("berry", "berries"), ("fry", "fries"),
                                 ("curry", "curries"), ("patty", "patties")]:
            assert fold_plural(singular) == fold_plural(plural)

    def test_ie_plurals_converge(self):
        from voxeval.metrics import fold_plural
        for singular, plural in [("cookie", "cookies"), ("pie", "pies")]:
            assert fold_plural(singular) == fold_plural(plural)

    def test_term_matching_finds_ies_plurals(self):
        from voxeval.metrics import Term, term_recall
        assert term_recall("a side of fries", [Term("fry")]).found == ["fry"]
        assert term_recall("mixed berries", [Term("berry")]).found == ["berry"]

    def test_still_not_prefix_matching(self):
        from voxeval.metrics import Term, term_recall
        assert not term_recall("a hamster wheel", [Term("ham")]).found
