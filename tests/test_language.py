from voxeval.metrics.language import (
    check_language,
    confidence_gate,
    detect_scripts,
    scripts_for_languages,
)


class TestDetectScripts:
    def test_detects_latin(self):
        assert detect_scripts("hello world") == {"latin"}

    def test_detects_han(self):
        assert "han" in detect_scripts("我要牛肉面")

    def test_ignores_digits_and_punctuation(self):
        # Script-neutral characters must not create false positives.
        assert detect_scripts("123 !@# ...") == set()

    def test_code_switching_reports_both(self):
        assert detect_scripts("I want 牛肉面") == {"latin", "han"}


class TestCheckLanguage:
    def test_english_transcript_on_english_agent_passes(self):
        assert check_language("I'd like to order a pizza", languages=["en"]).ok

    def test_flags_gujarati_on_english_agent(self):
        # Real incident: an English-only agent returned Gujarati for "egg roll",
        # so the item never reached the order.
        result = check_language("એંગ એકડો", languages=["en"])
        assert not result.ok
        assert "gujarati" in result.offending

    def test_flags_chinese_on_english_only_agent(self):
        assert not check_language("本来你不喜欢木钱的吗", languages=["en"]).ok

    def test_allows_chinese_on_multilingual_agent(self):
        assert check_language("我要牛肉面", languages=["en", "zh"]).ok

    def test_empty_transcript_passes(self):
        # Silence is not a hallucination.
        assert check_language("", languages=["en"]).ok

    def test_japanese_requires_han_and_kana(self):
        assert check_language("ラーメンを食べる", languages=["ja"]).ok

    def test_unknown_language_codes_are_ignored_not_raised(self):
        # An unrecognized code must not take down a deploy gate.
        assert scripts_for_languages(["en", "xx"]) == {"latin"}


class TestConfidenceGate:
    def test_missing_confidence_always_passes(self):
        # Providers that report nothing must not have every turn dropped.
        assert confidence_gate(None, 0.65)
        assert confidence_gate(0.0, 0.65)

    def test_threshold_of_zero_disables_the_gate(self):
        assert confidence_gate(0.1, 0.0)

    def test_drops_below_threshold_and_keeps_at_threshold(self):
        assert not confidence_gate(0.61, 0.65)
        assert confidence_gate(0.65, 0.65)
