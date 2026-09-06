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
