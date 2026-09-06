from voxeval.metrics.tools import (
    ToolCall,
    ToolExpectation,
    check_tool_calls,
    claimed_without_committing,
)


class TestToolCalls:
    def test_required_tool_called(self):
        result = check_tool_calls(
            [ToolCall("placeOrder"), ToolCall("confirmOrder")],
            ToolExpectation(required=("placeOrder", "confirmOrder")),
        )
        assert result.ok

    def test_missing_required_tool(self):
        result = check_tool_calls(
            [ToolCall("placeOrder")],
            ToolExpectation(required=("placeOrder", "confirmOrder")),
        )
        assert not result.ok
        assert result.missing == ["confirmOrder"]

    def test_forbidden_tool_called(self):
        # Closed-hours scenario: taking an order at 4am is the defect.
        result = check_tool_calls(
            [ToolCall("placeOrder")],
            ToolExpectation(forbidden=("placeOrder",)),
        )
        assert not result.ok
        assert result.forbidden_called == ["placeOrder"]

    def test_double_fired_commit_tool(self):
        # This is the one that double-charges customers.
        result = check_tool_calls(
            [ToolCall("placeOrder"), ToolCall("placeOrder")],
            ToolExpectation(commit_tools=("placeOrder",)),
        )
        assert not result.ok
        assert result.duplicated == ["placeOrder"]

    def test_non_commit_tool_may_repeat(self):
        # Lookups legitimately fire many times.
        result = check_tool_calls(
            [ToolCall("lookupMenu"), ToolCall("lookupMenu")],
            ToolExpectation(commit_tools=("placeOrder",)),
        )
        assert result.ok

    def test_argument_check_passes_on_any_matching_call(self):
        result = check_tool_calls(
            [ToolCall("book", {"date": "Monday"}), ToolCall("book", {"date": "Friday"})],
            ToolExpectation(argument_checks={"book": {"date": "Friday"}}),
        )
        assert result.ok

    def test_argument_check_fails_when_none_match(self):
        result = check_tool_calls(
            [ToolCall("book", {"date": "Monday"})],
            ToolExpectation(argument_checks={"book": {"date": "Friday"}}),
        )
        assert not result.ok
        assert result.argument_failures

    def test_argument_check_ignores_unspecified_keys(self):
        result = check_tool_calls(
            [ToolCall("book", {"date": "Friday", "party": 4, "notes": "window"})],
            ToolExpectation(argument_checks={"book": {"date": "Friday"}}),
        )
        assert result.ok


class TestFalseCompletion:
    def test_claimed_but_never_committed(self):
        # The words are exactly what success sounds like. Only the trace shows it.
        assert claimed_without_committing(
            "Great, your order is all set! You'll get a confirmation text.",
            [ToolCall("placeOrder")],
            commit_tools=("confirmOrder",),
        )

    def test_claimed_and_committed_is_fine(self):
        assert not claimed_without_committing(
            "Your order is all set!",
            [ToolCall("placeOrder"), ToolCall("confirmOrder")],
            commit_tools=("confirmOrder",),
        )

    def test_no_claim_no_problem(self):
        assert not claimed_without_committing(
            "We're closed right now, please call back tomorrow.",
            [],
            commit_tools=("confirmOrder",),
        )
