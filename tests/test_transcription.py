from voxeval.metrics.transcription import (
    Term,
    digit_accuracy,
    extract_digits,
    normalize,
    term_recall,
    word_error_rate,
)


class TestNormalize:
    def test_strips_punctuation_and_case(self):
        assert normalize("Kung-Pao, Chicken!") == "kung pao chicken"

    def test_normalizes_curly_apostrophes(self):
        # A curly apostrophe arriving straight is not a transcription error.
        # Counting it as one hides the errors that matter.
        assert normalize("General Tso’s") == normalize("General Tso's")

    def test_handles_none_and_empty(self):
        assert normalize("") == ""
        assert normalize(None) == ""


class TestWordErrorRate:
    def test_identical_is_zero(self):
        assert word_error_rate("hello there", "hello there") == 0.0

    def test_punctuation_and_case_are_not_errors(self):
        assert word_error_rate("Hello, there!", "hello there") == 0.0

    def test_single_substitution(self):
        assert word_error_rate("one two three", "one four three") == 1 / 3

    def test_deletion_and_insertion(self):
        assert word_error_rate("a b c", "a c") == 1 / 3
        assert word_error_rate("a c", "a b c") == 1 / 2

    def test_empty_reference(self):
        assert word_error_rate("", "") == 0.0
        assert word_error_rate("", "spurious words") == 1.0

    def test_can_exceed_one(self):
        # A hypothesis much longer than the reference legitimately scores above
        # 1.0. Callers must not assume this is a bounded percentage.
        assert word_error_rate("hi", "hi and then a great deal more text") > 1.0


class TestTermRecall:
    def test_finds_term_inside_longer_utterance(self):
        result = term_recall(
            "yeah I'd like one kung pao chicken please",
            [Term("Kung Pao Chicken", variants=("kung pao",))],
        )
        assert result.recall == 1.0
        assert result.missed == []

    def test_accepts_declared_variants(self):
        term = Term("Kung Pao Chicken", variants=("gong bao", "宫保"))
        assert term_recall("one gong bao chicken", [term]).recall == 1.0
        assert term_recall("我要宫保鸡丁", [term]).recall == 1.0

    def test_detects_known_collapse(self):
        # The motivating incident: low WER, but the item silently left the order.
        result = term_recall(
            "and a compound chicken please",
            [Term("Kung Pao Chicken", variants=("kung pao",), collapses=("compound chicken",))],
        )
        assert result.recall == 0.0
        assert result.has_collapse
        assert "compound chicken" in result.collapsed

    def test_partial_recall(self):
        terms = [Term("beef noodle"), Term("spring roll"), Term("fried rice")]
        result = term_recall("two beef noodle soups and three spring rolls", terms)
        assert result.recall == 2 / 3
        assert result.missed == ["fried rice"]

    def test_no_terms_is_full_recall(self):
        # Nothing required means nothing missed — must not divide by zero.
        assert term_recall("anything at all", []).recall == 1.0

    def test_collapse_without_matching_term_still_flagged(self):
        result = term_recall("compound chicken", [Term("x", collapses=("compound",))])
        assert result.has_collapse


class TestTermRecallWithReference:
    """One vocabulary passed to every case, scored only on what was actually said."""

    MENU = [
        Term("kung pao chicken", variants=("kung pao",), collapses=("compound chicken",)),
        Term("beef with broccoli", variants=("beef and broccoli",)),
    ]

    def test_only_spoken_terms_are_required(self):
        said = "I'd like one kung pao chicken, extra spicy."
        result = term_recall(said, self.MENU, reference=said)
        assert result.recall == 1.0
        assert result.found == ["kung pao chicken"]
        assert result.missed == []

    def test_a_spoken_term_can_still_be_missed(self):
        result = term_recall(
            "two beef with rockley please", self.MENU, reference="two beef with broccoli please"
        )
        assert result.recall == 0.0
        assert result.missed == ["beef with broccoli"]

    def test_variant_in_reference_makes_the_term_required(self):
        result = term_recall("beef and broccoli", self.MENU, reference="beef and broccoli")
        assert result.found == ["beef with broccoli"]

    def test_no_spoken_terms_is_unmeasured_not_missed(self):
        # The live-run bug: a perfect "Yes." scored 0/2 against the menu.
        result = term_recall("Yes.", self.MENU, reference="Yes.")
        assert result.missed == []
        assert not result.measured

    def test_collapse_still_caught_when_term_was_spoken(self):
        result = term_recall(
            "one compound chicken", self.MENU, reference="one kung pao chicken"
        )
        assert result.missed == ["kung pao chicken"]
        assert result.collapsed == ["compound chicken"]

    def test_collapse_the_speaker_said_is_not_a_mishear(self):
        result = term_recall(
            "is compound chicken a dish", self.MENU, reference="is compound chicken a dish"
        )
        assert result.collapsed == []

    def test_accented_term_spoken_in_other_case_is_still_required(self):
        # A spoken term must never drop out of scoring because of case or Unicode form.
        result = term_recall("one coffee please", [Term("café")], reference="one CAFÉ please")
        assert result.missed == ["café"]
        assert result.measured

    def test_accented_term_matches_hypothesis_case_insensitively(self):
        decomposed = "cafe\u0301"  # "café" as e + combining accent
        result = term_recall(f"one {decomposed.upper()} please", [Term("café")], reference="one café please")
        assert result.found == ["café"]

    def test_devanagari_vowel_signs_are_not_erased(self):
        # दिन (day) and दीन (poor) differ only in a combining vowel sign.
        result = term_recall("दीन", [Term("दिन")], reference="दिन")
        assert result.missed == ["दिन"]

    def test_fullwidth_latin_keeps_token_boundaries(self):
        # "ｔｅａ" folds to "tea", which must not match inside "steak".
        result = term_recall("steak", [Term("ｔｅａ")], reference="steak")
        assert not result.measured
        assert term_recall("one ｔｅａ please", [Term("ｔｅａ")], reference="one tea please").found == ["ｔｅａ"]

    def test_cjk_term_still_matches_inside_unsegmented_text(self):
        term = Term("kung pao chicken", variants=("宫保鸡丁",))
        result = term_recall("我要宫保鸡丁", [term], reference="我要宫保鸡丁")
        assert result.found == ["kung pao chicken"]

    def test_without_reference_every_term_is_required(self):
        # Unchanged: with no reference there is no way to know what was said.
        result = term_recall("Yes.", self.MENU)
        assert result.recall == 0.0
        assert result.measured


class TestDigits:
    def test_extracts_spelled_out_numbers(self):
        assert extract_digits("six one seven five five five") == "617555"

    def test_extracts_numerals(self):
        assert extract_digits("call 617-555-0142") == "6175550142"

    def test_treats_oh_as_zero(self):
        assert extract_digits("five oh two") == "502"

    def test_spoken_and_numeric_forms_are_equivalent(self):
        # WER would score these as completely different; they are the same number.
        assert digit_accuracy("five one seven", "517") == 1.0

    def test_wrong_digit_is_penalized(self):
        assert digit_accuracy("617555", "617554") < 1.0

    def test_no_digits_in_reference_passes(self):
        assert digit_accuracy("no numbers here", "none here either") == 1.0

    def test_missing_all_digits_scores_zero(self):
        assert digit_accuracy("617", "") == 0.0


class TestTermMatchingIsTokenBounded:
    """Substring matching reported Term("ham") as found in "champagne" — a false
    positive on the metric whose whole job is proving the transaction words
    survived. A false 'found' is worse than a miss: it reads as success."""

    def test_short_term_does_not_match_inside_a_longer_word(self):
        result = term_recall("a glass of champagne", [Term("ham")])
        assert result.missed == ["ham"]

    def test_term_still_matches_inside_a_longer_utterance(self):
        result = term_recall(
            "yes I want the beef with broccoli please", [Term("beef with broccoli")])
        assert result.found == ["beef with broccoli"]

    def test_exact_single_token(self):
        assert term_recall("one ham sandwich", [Term("ham")]).found == ["ham"]

    def test_collapse_detection_is_also_token_bounded(self):
        term = Term("kung pao chicken", collapses=("compound chicken",))
        assert not term_recall("compounding chicken stock", [term]).collapsed
        assert term_recall("compound chicken please", [term]).collapsed

    def test_non_ascii_keeps_substring_matching(self):
        # CJK has no whitespace token boundaries; "牛肉面" is one word to a reader
        # and several tokens to .split().
        term = Term("beef noodle soup", variants=("牛肉面",))
        assert term_recall("我想要牛肉面外带", [term]).found == ["beef noodle soup"]

    def test_multiword_term_must_be_contiguous(self):
        result = term_recall("beef and also broccoli", [Term("beef broccoli")])
        assert result.missed == ["beef broccoli"]


class TestPluralFolding:
    """Token-boundary matching must not break the commonest legitimate variation.
    Folding is symmetric, so it need not be correct English — only consistent."""

    def test_regular_plural(self):
        assert term_recall("two spring rolls", [Term("spring roll")]).found

    def test_singular_hypothesis_plural_term(self):
        assert term_recall("one spring roll", [Term("spring rolls")]).found

    def test_es_plural_after_sibilant(self):
        for hyp, term in [("two glasses", "glass"), ("three boxes", "box"),
                          ("two dishes", "dish"), ("two sandwiches", "sandwich")]:
            assert term_recall(hyp, [Term(term)]).found, f"{term} in {hyp}"

    def test_es_rule_does_not_over_strip(self):
        # "roses" must fold to "rose", not "ros" — the stem has to end in a real
        # sibilant (ss/x/z/ch/sh), not a bare s.
        assert term_recall("a dozen roses", [Term("rose")]).found

    def test_folding_does_not_reintroduce_substring_matching(self):
        assert not term_recall("a hamster wheel", [Term("ham")]).found
        assert not term_recall("a glass of champagne", [Term("ham")]).found
