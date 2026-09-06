"""LLM judge: verdicts, failure isolation, and prompt-reply parsing."""

from __future__ import annotations

from voxeval import ConversationCase, Suite
from voxeval.judges import _extract_json, llm_judge
from voxeval.metrics.judge import JudgeReport, Verdict, safe_judge


class TestVerdict:
    def test_passed_defaults_from_score(self):
        assert Verdict(score=85).passed
        assert not Verdict(score=42).passed

    def test_explicit_passed_overrides(self):
        # A rubric may have hard disqualifiers — inventing a price fails at any
        # score.
        assert not Verdict(score=95, passed=False).passed

    def test_errored_verdict(self):
        assert Verdict(score=0, error="boom").errored


class TestSafeJudge:
    def test_exception_becomes_errored_verdict(self):
        def broken(scenario, expected, transcript):
            raise RuntimeError("provider down")

        verdict = safe_judge(broken, "s", "e", "t")
        assert verdict.errored
        # Must not fail the agent: the provider being down says nothing about it.
        assert verdict.passed

    def test_bare_float_is_wrapped(self):
        assert safe_judge(lambda s, e, t: 91.0, "s", "e", "t").score == 91.0

    def test_verdict_passes_through(self):
        v = Verdict(score=50, reasoning="thin")
        assert safe_judge(lambda s, e, t: v, "s", "e", "t") is v


class TestJudgeReport:
    def test_errors_excluded_from_mean(self):
        # The core aggregation rule: averaging a failed API call in as a zero
        # makes a provider outage look like an agent regression.
        report = JudgeReport([
            Verdict(score=90), Verdict(score=80), Verdict(score=0, error="timeout"),
        ])
        assert report.mean_score == 85.0
        assert report.errors == 1

    def test_all_errored_reports_no_score(self):
        report = JudgeReport([Verdict(score=0, error="x")])
        assert report.mean_score is None
        assert report.pass_rate is None

    def test_empty(self):
        assert JudgeReport().mean_score is None


class TestLlmJudge:
    def test_parses_a_well_formed_reply(self):
        judge = llm_judge(lambda p: '{"observations": "good", "score": 88, "abstain": false}')
        verdict = judge("order a coffee", "takes the order", "caller: coffee. agent: got it.")
        assert verdict.score == 88
        assert verdict.passed
        assert verdict.reasoning == "good"

    def test_tolerates_fenced_json(self):
        judge = llm_judge(lambda p: '```json\n{"score": 75}\n```')
        assert judge("s", "e", "t").score == 75

    def test_tolerates_prose_around_json(self):
        judge = llm_judge(lambda p: 'Here is my assessment:\n{"score": 60}\nHope that helps.')
        assert judge("s", "e", "t").score == 60

    def test_abstention_is_an_error_not_a_zero(self):
        judge = llm_judge(lambda p: '{"score": 0, "abstain": true}')
        verdict = judge("s", "e", "t")
        assert verdict.errored and verdict.passed

    def test_unparseable_reply_is_an_error_not_a_zero(self):
        # A fault in the judge must not be charged to the agent, or a model
        # upgrade reads as a regression.
        judge = llm_judge(lambda p: "I think it was pretty good actually")
        verdict = judge("s", "e", "t")
        assert verdict.errored and verdict.passed

    def test_missing_score_field(self):
        assert llm_judge(lambda p: '{"observations": "hm"}')("s", "e", "t").errored

    def test_empty_transcript_is_not_judged(self):
        called = []
        judge = llm_judge(lambda p: called.append(p) or '{"score": 100}')
        verdict = judge("s", "e", "   ")
        assert verdict.errored
        assert not called, "should not spend an API call on an empty transcript"

    def test_score_is_clamped(self):
        assert llm_judge(lambda p: '{"score": 150}')("s", "e", "t").score == 100.0
        assert llm_judge(lambda p: '{"score": -20}')("s", "e", "t").score == 0.0

    def test_prompt_carries_scenario_expectation_and_transcript(self):
        seen = {}
        judge = llm_judge(lambda p: seen.setdefault("p", p) and '{"score": 80}')
        judge("ORDER_SCENARIO", "EXPECTED_RUBRIC", "TRANSCRIPT_TEXT")
        for token in ("ORDER_SCENARIO", "EXPECTED_RUBRIC", "TRANSCRIPT_TEXT"):
            assert token in seen["p"]

    def test_custom_pass_score(self):
        judge = llm_judge(lambda p: '{"score": 75}', pass_score=90)
        assert not judge("s", "e", "t").passed


class TestExtractJson:
    def test_returns_none_for_non_object(self):
        assert _extract_json("[1, 2, 3]") is None
        assert _extract_json("") is None

    def test_handles_nested_braces(self):
        assert _extract_json('text {"a": {"b": 1}} tail') == {"a": {"b": 1}}


class TestJudgeInSuite:
    def _case(self, **kw):
        return ConversationCase(id="c1", transcript="caller: hi. agent: hello.", **kw)

    def test_not_judged_without_a_rubric(self):
        # A judge with no rubric produces a number that drifts with the model
        # version and measures nothing.
        suite = Suite().add_conversation(self._case(judge=lambda s, e, t: 90.0))
        assert suite.run().metrics["judge_score"] is None

    def test_judged_when_rubric_present(self):
        suite = Suite().add_conversation(
            self._case(judge=lambda s, e, t: 90.0, expected_behavior="greets the caller")
        )
        assert suite.run().metrics["judge_score"] == 90.0

    def test_unjudged_suite_reports_none_not_zero(self):
        # Scoring an unconfigured judge as 0 would fail every suite that never
        # opted in.
        assert Suite().add_conversation(self._case()).run().metrics["judge_score"] is None

    def test_low_score_does_not_fail_the_case_boolean(self):
        report = Suite().add_conversation(
            self._case(judge=lambda s, e, t: 10.0, expected_behavior="greets")
        ).run()
        assert report.conversations[0].ok, "judge is advisory, not a per-case failure"

    def test_low_score_warns_but_does_not_block_the_gate(self):
        report = Suite().add_conversation(
            self._case(judge=lambda s, e, t: 10.0, expected_behavior="greets")
        ).run()
        result = report.gate()
        assert result.passed, "judge variance must not block a deploy by default"
        assert any(w.threshold.name == "judge_score" for w in result.warnings)

    def test_judge_crash_does_not_take_down_the_run(self):
        def broken(s, e, t):
            raise RuntimeError("429")

        report = Suite().add_conversation(
            self._case(judge=broken, expected_behavior="greets")
        ).run()
        assert report.metrics["judge_score"] is None
        assert report.judge is not None and report.judge.errors == 1
        assert report.gate().passed


class TestRubricGuidance:
    """Encodes the two rubric failures found running this against a live agent."""

    def test_judge_sees_only_what_it_is_given(self):
        # A rubric asking whether a tool fired, with a transcript that omits tool
        # lines, made the judge infer from wording and score a correct call
        # 40/100. The library cannot stop you writing that rubric, but the
        # template must not encourage the inference.
        from voxeval.judges import DEFAULT_TEMPLATE
        assert "Do not assume the agent called a tool" in DEFAULT_TEMPLATE

    def test_template_offers_abstention(self):
        # Without an explicit way to say "I cannot tell", a judge invents a
        # judgement, and an invented one is indistinguishable from a real one.
        from voxeval.judges import DEFAULT_TEMPLATE
        assert "abstain" in DEFAULT_TEMPLATE


class TestAbstainMustBeBoolean:
    def test_string_false_is_not_an_abstention(self):
        # Raw truthiness read "false" as an abstention, and abstentions are
        # dropped from the mean — so a wrong-typed field silently removed cases
        # from the score.
        from voxeval.judges import llm_judge
        verdict = llm_judge(lambda p: '{"score": 88, "abstain": "false"}')("s", "e", "t")
        assert verdict.errored, "non-boolean abstain is a malformed reply"

    def test_real_boolean_abstain_still_works(self):
        from voxeval.judges import llm_judge
        assert llm_judge(lambda p: '{"score": 0, "abstain": true}')("s", "e", "t").errored

    def test_missing_abstain_defaults_to_scoring(self):
        from voxeval.judges import llm_judge
        verdict = llm_judge(lambda p: '{"score": 88}')("s", "e", "t")
        assert not verdict.errored and verdict.score == 88
