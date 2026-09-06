"""End-to-end: cases → suite → aggregated metrics → gate decision."""

from voxeval import (
    ConversationCase,
    Suite,
    Term,
    ToolCall,
    ToolExpectation,
    TurnCase,
    TurnLatency,
    Utterance,
)


def _healthy_suite() -> Suite:
    suite = Suite("healthy")
    suite.add_turn(TurnCase(
        id="order",
        reference="I'd like one kung pao chicken and two spring rolls",
        hypothesis="I'd like one kung pao chicken and two spring rolls",
        terms=[Term("Kung Pao Chicken", variants=("kung pao",), collapses=("compound",)),
               Term("spring roll")],
        latency=TurnLatency(vad_to_stt_ms=180, stt_to_llm_ms=420, llm_to_tts_ms=90),
        languages=["en"],
    ))
    suite.add_conversation(ConversationCase(
        id="takeout",
        transcript="Your order is all set, you'll get a confirmation text.",
        tool_calls=[ToolCall("placeOrder"), ToolCall("confirmOrder")],
        expected_tools=ToolExpectation(
            required=("placeOrder", "confirmOrder"),
            commit_tools=("placeOrder", "confirmOrder"),
        ),
        utterances=[Utterance("caller", 0, 1500), Utterance("agent", 1800, 3200)],
    ))
    return suite


class TestSuiteEndToEnd:
    def test_healthy_run_passes_the_gate(self):
        report = _healthy_suite().run()
        assert report.gate().passed
        assert not report.failed_turns
        assert not report.failed_conversations

    def test_metrics_are_aggregated(self):
        metrics = _healthy_suite().run().metrics
        assert metrics["term_recall"] == 1.0
        assert metrics["wer"] == 0.0
        assert metrics["collapses"] == 0
        assert metrics["end_to_end_p50_ms"] == 690  # 180 + 420 + 90

    def test_collapse_fails_the_gate_despite_low_wer(self):
        """The headline case: 'compound chicken' for 'kung pao chicken'.

        WER stays low because most words are right. The item silently leaves the
        order. A WER-only gate ships this.
        """
        suite = Suite()
        suite.add_turn(TurnCase(
            id="collapse",
            reference="I want one kung pao chicken extra spicy please",
            hypothesis="I want one compound chicken extra spicy please",
            terms=[Term("Kung Pao Chicken", variants=("kung pao",), collapses=("compound chicken",))],
        ))
        report = suite.run()

        assert report.metrics["wer"] < 0.25, "WER alone looks acceptable"
        assert report.metrics["collapses"] == 1
        assert not report.gate().passed
        assert report.failed_turns

    def test_false_completion_fails_the_gate(self):
        """Agent says the order is placed but never commits it."""
        suite = Suite()
        suite.add_conversation(ConversationCase(
            id="phantom",
            transcript="Great, your order is all set! See you at 6.",
            tool_calls=[ToolCall("placeOrder")],  # drafted, never confirmed
            expected_tools=ToolExpectation(commit_tools=("confirmOrder",)),
        ))
        report = suite.run()
        assert report.metrics["claimed_without_commit"] == 1
        assert not report.gate().passed

    def test_out_of_set_language_fails_the_gate(self):
        suite = Suite()
        suite.add_turn(TurnCase(id="halluc", hypothesis="એંગ એકડો", languages=["en"]))
        report = suite.run()
        assert report.metrics["out_of_set_language_rate"] == 1.0
        assert not report.gate().passed

    def test_duplicate_commit_fails_the_gate(self):
        suite = Suite()
        suite.add_conversation(ConversationCase(
            id="double",
            transcript="All set.",
            tool_calls=[ToolCall("placeOrder"), ToolCall("placeOrder"), ToolCall("confirmOrder")],
            expected_tools=ToolExpectation(commit_tools=("placeOrder",)),
        ))
        assert not suite.run().gate().passed

    def test_empty_suite_does_not_crash_or_block(self):
        report = Suite().run()
        assert report.gate().passed

    def test_report_renders(self):
        assert "order" in str(_healthy_suite().run())


class TestDigitTurnsDoNotPoisonWer:
    """A spoken phone number transcribed perfectly as numerals scores WER 1.0.

    Left in the aggregate it drags a healthy suite toward the threshold and
    eventually fails a deploy over a turn that was exactly right.
    """

    def _case(self, score_wer: bool) -> TurnCase:
        return TurnCase(
            id="callback",
            reference="my number is six one seven five five five",
            hypothesis="my number is 617 555",
            score_wer=score_wer,
        )

    def test_included_by_default_and_scores_badly(self):
        report = Suite().add_turn(self._case(True)).run()
        assert report.metrics["wer"] > 0.5

    def test_opting_out_removes_it_from_the_wer_aggregate(self):
        report = Suite().add_turn(self._case(False)).run()
        assert report.metrics["wer"] is None

    def test_digit_accuracy_still_scores_when_wer_is_suppressed(self):
        report = Suite().add_turn(self._case(False)).run()
        assert report.metrics["digit_accuracy"] == 1.0


class TestToolFailuresReachTheGate:
    """Regression: tool-call failures were computed per conversation but never
    aggregated, so a run that never called the commit tool printed 'never called
    confirmOrder' in the detail and still exited 0."""

    def test_missing_required_tool_blocks_the_deploy(self):
        suite = Suite()
        suite.add_conversation(ConversationCase(
            id="never-confirmed",
            transcript="Order placed.",
            tool_calls=[ToolCall("placeOrder")],
            expected_tools=ToolExpectation(required=("placeOrder", "confirmOrder")),
        ))
        report = suite.run()
        assert report.metrics["tool_call_failures"] == 1
        assert not report.gate().passed

    def test_forbidden_tool_blocks_the_deploy(self):
        suite = Suite()
        suite.add_conversation(ConversationCase(
            id="closed-hours",
            transcript="Sure, I'll place that.",
            tool_calls=[ToolCall("placeOrder")],
            expected_tools=ToolExpectation(forbidden=("placeOrder",)),
        ))
        assert not suite.run().gate().passed

    def test_correct_trace_does_not_block(self):
        suite = Suite()
        suite.add_conversation(ConversationCase(
            id="clean",
            transcript="All set.",
            tool_calls=[ToolCall("placeOrder"), ToolCall("confirmOrder")],
            expected_tools=ToolExpectation(required=("placeOrder", "confirmOrder")),
        ))
        report = suite.run()
        assert report.metrics["tool_call_failures"] == 0
        assert report.gate().passed


class TestMetricKeysAreStable:
    """Keys that appear only when measured make report.metrics[...] raise for one
    run and work for the next. The gate tolerates it via .get(); users don't."""

    EXPECTED = {
        "wer", "term_recall", "collapses", "digit_accuracy", "out_of_set_language_rate",
        "end_to_end_p50_ms", "end_to_end_p95_ms", "stt_p95_ms", "llm_ttft_p95_ms",
        "tts_ttfa_p95_ms", "barge_in_recovery_p95_ms", "tool_call_failures",
        "duplicate_commits", "claimed_without_commit", "dead_air_events",
        "semantic_wer", "semantic_similarity", "judge_score", "judge_pass_rate",
    }

    def test_empty_suite_still_has_every_key(self):
        assert set(Suite().run().metrics) == self.EXPECTED

    def test_populated_suite_has_the_same_keys(self):
        assert set(_healthy_suite().run().metrics) == self.EXPECTED

    def test_unmeasured_metrics_are_none_not_missing(self):
        metrics = Suite().add_turn(TurnCase(id="t", hypothesis="hi")).run().metrics
        assert metrics["stt_p95_ms"] is None
        assert metrics["wer"] is None


class TestVoiceMultiTurn:
    """A Turn is the same object for voice and text; only the transcription gap
    between `spoken` and `heard` distinguishes them."""

    def _voice_call(self):
        from voxeval import ConversationCase, Term, Turn
        return ConversationCase(id="order", turns=[
            Turn(spoken="I want one kung pao chicken",
                 heard="I want one compound chicken",
                 agent_said="Sorry, we don't have that.",
                 terms=[Term("kung pao chicken", collapses=("compound chicken",))]),
            Turn(spoken="the kung pao chicken please",
                 heard="the kung pao chicken please",
                 agent_said="Got it.",
                 terms=[Term("kung pao chicken")]),
        ])

    def test_transcript_is_derived_from_turns(self):
        case = self._voice_call()
        assert "Customer:" in case.transcript and "Agent:" in case.transcript

    def test_transcript_uses_what_was_heard_not_what_was_said(self):
        # The agent replied to "compound chicken"; judging it against words it
        # never received would score the recognizer twice.
        assert "compound chicken" in self._voice_call().transcript
        assert "one kung pao chicken" not in self._voice_call().transcript

    def test_nested_turns_reach_the_suite_metrics(self):
        from voxeval import Suite
        report = Suite().add_conversation(self._voice_call()).run()
        assert report.metrics["wer"] is not None, "turn metrics must not vanish"
        assert report.metrics["collapses"] == 1.0

    def test_collapse_in_a_turn_fails_the_conversation(self):
        from voxeval import Suite
        report = Suite().add_conversation(self._voice_call()).run()
        assert not report.conversations[0].ok

    def test_text_mode_turns_carry_no_transcription_gap(self):
        from voxeval import ConversationCase, Suite, Turn
        case = ConversationCase(id="text", turns=[
            Turn(spoken="book me a table", agent_said="Sure, for when?"),
        ])
        report = Suite().add_conversation(case).run()
        # Same object, same pipeline — WER simply has nothing to compare.
        assert report.metrics["wer"] is None
        assert "book me a table" in case.transcript

    def test_tool_calls_and_latency_derive_from_turns(self):
        from voxeval import ConversationCase, ToolCall, Turn, TurnLatency
        case = ConversationCase(id="c", turns=[
            Turn(spoken="order it", tool_calls=[ToolCall("placeOrder")],
                 latency=TurnLatency(end_to_end_ms=800.0)),
        ])
        assert [c.name for c in case.tool_calls] == ["placeOrder"]
        assert len(case.latencies) == 1

    def test_explicit_fields_are_not_overwritten(self):
        from voxeval import ConversationCase, Turn
        case = ConversationCase(id="c", transcript="explicit",
                                turns=[Turn(spoken="derived")])
        assert case.transcript == "explicit"

    def test_flat_transcript_still_works(self):
        from voxeval import ConversationCase, Suite
        case = ConversationCase(id="legacy", transcript="Customer: hi\nAgent: hello")
        assert Suite().add_conversation(case).run().conversations[0].ok


class TestExplicitEmptyIsNotAbsent:
    """`[]` is a claim, not an absence.

    Deriving over an explicitly empty list means the library silently
    contradicts its own input on the one metric where being wrong means a
    customer got charged.
    """

    def _turns(self):
        from voxeval import ToolCall, Turn, TurnLatency
        return [Turn(spoken="place it",
                     tool_calls=[ToolCall("confirmOrder")],
                     latency=TurnLatency(end_to_end_ms=500.0))]

    def test_explicit_empty_tool_calls_survives(self):
        from voxeval import ConversationCase
        case = ConversationCase(id="c", turns=self._turns(), tool_calls=[])
        assert case.tool_calls == []

    def test_omitted_tool_calls_derives_from_turns(self):
        from voxeval import ConversationCase
        case = ConversationCase(id="c", turns=self._turns())
        assert [c.name for c in case.tool_calls] == ["confirmOrder"]

    def test_explicit_empty_does_not_satisfy_a_required_tool(self):
        # The exact failure: a caller asserts no tool fired, the turns contain
        # one, and a `required` check passes on data the caller said was empty.
        from voxeval import ConversationCase, Suite, ToolExpectation
        report = Suite().add_conversation(ConversationCase(
            id="c", turns=self._turns(), tool_calls=[],
            expected_tools=ToolExpectation(required=("confirmOrder",)),
        )).run()
        assert report.metrics["tool_call_failures"] == 1.0

    def test_explicit_empty_latencies_survives(self):
        from voxeval import ConversationCase
        assert ConversationCase(id="c", turns=self._turns(), latencies=[]).latencies == []

    def test_explicit_transcript_wins_over_turns(self):
        from voxeval import ConversationCase
        case = ConversationCase(id="c", turns=self._turns(), transcript="explicit")
        assert case.transcript == "explicit"

    def test_no_turns_and_no_transcript_yields_empty_string(self):
        from voxeval import ConversationCase
        assert ConversationCase(id="c").transcript == ""
