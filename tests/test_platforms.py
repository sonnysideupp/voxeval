"""Platform payload adapters.

Fixtures follow each vendor's published webhook shape. They have NOT been run
against a live account, so these tests pin the mapping, not the vendor.
"""

from __future__ import annotations

import pytest

from voxeval.platforms import (
    EmptyPayloadError,
    from_openai_messages,
    from_retell,
    from_vapi,
)


class TestOpenAIMessages:
    def test_turns_are_assembled_from_alternating_roles(self):
        trace = from_openai_messages([
            {"role": "user", "content": "one kung pao chicken"},
            {"role": "assistant", "content": "Got it. Anything else?"},
            {"role": "user", "content": "that's all"},
            {"role": "assistant", "content": "All set."},
        ])
        assert len(trace.turns) == 2
        assert trace.turns[0].heard == "one kung pao chicken"
        assert trace.turns[0].agent_said == "Got it. Anything else?"

    def test_tool_calls_attach_to_the_preceding_caller_turn(self):
        # Per-turn attribution is what makes "claimed completion on the turn it
        # never called the tool" detectable, rather than only per-call.
        trace = from_openai_messages([
            {"role": "user", "content": "book it"},
            {"role": "assistant", "content": "Booking.",
             "tool_calls": [{"function": {"name": "placeOrder",
                                          "arguments": '{"item": "soup"}'}}]},
        ])
        assert [c.name for c in trace.turns[0].tool_calls] == ["placeOrder"]
        assert trace.turns[0].tool_calls[0].arguments == {"item": "soup"}

    def test_arguments_accepted_as_dict_or_json_string(self):
        for args in ('{"a": 1}', {"a": 1}):
            trace = from_openai_messages([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [
                    {"function": {"name": "f", "arguments": args}}]},
            ])
            assert trace.turns[0].tool_calls[0].arguments == {"a": 1}

    def test_malformed_arguments_degrade_to_empty(self):
        trace = from_openai_messages([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "f", "arguments": "not json"}}]},
        ])
        assert trace.turns[0].tool_calls[0].arguments == {}

    def test_agent_only_call_still_produces_a_trace(self):
        # Voicemail or an immediate hangup. Still judgeable.
        trace = from_openai_messages([
            {"role": "assistant", "content": "Thanks for calling, we're closed."},
        ])
        assert "closed" in trace.turns[0].agent_said

    def test_empty_messages_raise_rather_than_scoring_clean(self):
        # An empty trace scores as a perfect conversation — no failed
        # assertions, nothing to judge. A dropped webhook must not read as a
        # passing call.
        with pytest.raises(EmptyPayloadError):
            from_openai_messages([])

    def test_spoken_is_none_because_platforms_return_only_what_was_heard(self):
        # No ground truth exists in a hosted trace, so transcription metrics
        # must not silently score the recognizer against itself.
        trace = from_openai_messages([{"role": "user", "content": "hi"}])
        assert trace.turns[0].spoken is None
        assert not trace.turns[0].is_voice


class TestRetell:
    PAYLOAD = {
        "event": "call_ended",
        "call": {
            "call_id": "abc123",
            "transcript": "User: one soup\nAgent: got it",
            "transcript_object": [
                {"role": "user", "content": "one soup"},
                {"role": "agent", "content": "got it"},
            ],
            "transcript_with_tool_calls": [
                {"role": "user", "content": "one soup"},
                {"role": "agent", "content": "placing that now"},
                {"role": "tool_call_invocation", "tool_call_id": "t1",
                 "name": "place_order", "arguments": '{"item": "soup"}'},
                {"role": "tool_call_result", "tool_call_id": "t1", "content": "ok"},
                {"role": "agent", "content": "all set"},
            ],
        },
    }

    def test_prefers_the_view_that_contains_tool_calls(self):
        # transcript_object drops tool calls silently, which would make every
        # tool assertion pass vacuously.
        trace = from_retell(self.PAYLOAD)
        assert [c.name for c in trace.turns[0].tool_calls] == ["place_order"]

    def test_tool_result_is_not_counted_as_a_second_call(self):
        # Counting the result would double every call and trip duplicate-commit.
        trace = from_retell(self.PAYLOAD)
        assert len(trace.turns[0].tool_calls) == 1

    def test_agent_utterances_are_joined(self):
        trace = from_retell(self.PAYLOAD)
        assert "placing that now" in trace.turns[0].agent_said
        assert "all set" in trace.turns[0].agent_said

    def test_falls_back_to_transcript_object(self):
        payload = {"call": {"transcript_object": self.PAYLOAD["call"]["transcript_object"]}}
        trace = from_retell(payload)
        assert trace.turns[0].heard == "one soup"

    def test_flat_transcript_last_resort(self):
        trace = from_retell({"call": {"transcript": "User: hi\nAgent: hello"}})
        assert "hello" in trace.transcript
        assert not trace.turns

    def test_unwrapped_payload_also_works(self):
        trace = from_retell({"transcript_object": [{"role": "user", "content": "hi"}]})
        assert trace.turns[0].heard == "hi"

    def test_empty_payload_raises(self):
        with pytest.raises(EmptyPayloadError, match="retell"):
            from_retell({})


class TestVapi:
    def test_end_of_call_report_with_artifact_messages(self):
        trace = from_vapi({"message": {
            "type": "end-of-call-report",
            "artifact": {"messages": [
                {"role": "user", "content": "one soup"},
                {"role": "assistant", "content": "placing it", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "placeOrder", "arguments": {"item": "soup"}}}]},
            ]},
        }})
        assert trace.turns[0].heard == "one soup"
        assert [c.name for c in trace.turns[0].tool_calls] == ["placeOrder"]

    def test_top_level_tool_calls_are_merged(self):
        trace = from_vapi({"message": {
            "artifact": {"messages": [{"role": "user", "content": "hi"}]},
            "toolCalls": [{"id": "c1", "type": "function",
                           "function": {"name": "lookup", "arguments": {}}}],
        }})
        assert [c.name for c in trace.turns[0].tool_calls] == ["lookup"]

    def test_duplicate_tool_calls_are_not_double_counted(self):
        # The same call appearing inline and at top level must not read as two —
        # that would trip duplicate-commit on a healthy agent.
        call = {"id": "c1", "type": "function",
                "function": {"name": "placeOrder", "arguments": {"item": "soup"}}}
        trace = from_vapi({"message": {
            "artifact": {"messages": [
                {"role": "user", "content": "order it"},
                {"role": "assistant", "content": "ok", "tool_calls": [call]},
            ]},
            "toolCalls": [call],
        }})
        assert len(trace.turns[0].tool_calls) == 1

    def test_transcript_only_payload(self):
        trace = from_vapi({"message": {"transcript": "User: hi\nAssistant: hello"}})
        assert "hello" in trace.transcript

    def test_empty_payload_raises(self):
        with pytest.raises(EmptyPayloadError, match="vapi"):
            from_vapi({})


class TestEndToEndScoring:
    def test_a_platform_trace_scores_through_the_normal_suite(self):
        from voxeval import ConversationCase, Suite, ToolExpectation

        trace = from_retell(TestRetell.PAYLOAD)
        report = Suite().add_conversation(ConversationCase(
            id="call-abc123", turns=trace.turns,
            expected_tools=ToolExpectation(required=("place_order",),
                                           commit_tools=("place_order",)),
        )).run()
        assert report.metrics["tool_call_failures"] == 0.0
        assert report.metrics["duplicate_commits"] == 0.0
        # No ground truth in a hosted trace, so transcription is not scored.
        assert report.metrics["wer"] is None

    def test_false_completion_is_caught_in_a_platform_trace(self):
        from voxeval import ConversationCase, Suite, ToolExpectation

        trace = from_openai_messages([
            {"role": "user", "content": "place my order"},
            {"role": "assistant", "content": "You're all set, it'll be ready soon."},
        ])
        report = Suite().add_conversation(ConversationCase(
            id="phantom", turns=trace.turns,
            expected_tools=ToolExpectation(required=("placeOrder",),
                                           commit_tools=("placeOrder",)),
        )).run()
        assert report.metrics["claimed_without_commit"] == 1.0


class TestVapiNativeMessageShape:
    """Vapi's artifact.messages stores text under `message`, not `content`.

    Handing it straight to an OpenAI-shaped parser yields a turn per utterance
    with empty text: structurally a conversation, semantically gone. It slips
    past the empty-payload guard because turns exist.
    """

    DOCUMENTED = {
        "message": {
            "type": "end-of-call-report",
            "artifact": {"messages": [
                {"role": "system", "message": "You are a helpful assistant..."},
                {"role": "assistant", "message": "How can I help?"},
                {"role": "user", "message": "One kung pao chicken please"},
                {"role": "assistant", "message": "Placing that now.",
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "placeOrder",
                                              "arguments": {"item": "kung pao"}}}]},
            ]},
        },
    }

    def test_native_message_field_is_read(self):
        trace = from_vapi(self.DOCUMENTED)
        assert trace.turns, "documented payload must produce turns"
        assert trace.turns[0].heard == "One kung pao chicken please"

    def test_turns_are_not_empty(self):
        trace = from_vapi(self.DOCUMENTED)
        assert any(t.received for t in trace.turns)

    def test_tool_calls_survive_normalization(self):
        trace = from_vapi(self.DOCUMENTED)
        assert [c.name for c in trace.turns[0].tool_calls] == ["placeOrder"]

    def test_system_prompt_is_not_treated_as_speech(self):
        # Vapi includes the full system prompt in artifact.messages.
        trace = from_vapi(self.DOCUMENTED)
        assert not any("helpful assistant" in (t.received or "") for t in trace.turns)

    def test_openai_formatted_is_preferred_when_present(self):
        payload = {"message": {"artifact": {
            "messages": [{"role": "user", "message": "native"}],
            "messagesOpenAIFormatted": [{"role": "user", "content": "openai"}],
        }}}
        assert from_vapi(payload).turns[0].heard == "openai"

    def test_all_empty_turns_raise_rather_than_score_clean(self):
        # The guard for the failure this class exists to document: if the text
        # field is ever renamed again, fail loudly instead of reporting a call
        # in which nobody spoke.
        with pytest.raises(EmptyPayloadError, match="every one is empty"):
            from_openai_messages([
                {"role": "user", "mystery_field": "hello"},
                {"role": "assistant", "mystery_field": "hi"},
            ])
