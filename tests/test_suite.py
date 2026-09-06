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
