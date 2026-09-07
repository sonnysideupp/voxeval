"""Turn a hosted platform's call payload into something scoreable.

The library scores traces. Getting a trace out of a hosted voice platform is
mechanical but fiddly — every vendor names the same three things differently —
so the mapping lives here instead of in every user's harness.

    from voxeval import Suite, ConversationCase
    from voxeval.platforms import from_retell

    trace = from_retell(webhook_payload)          # your call_ended webhook
    suite.add_conversation(ConversationCase(
        id=call_id, turns=trace.turns, tool_calls=trace.tool_calls,
        expected_tools=..., expected_behavior=...))

**What these adapters do and do not give you.** They recover the conversation and
the tool trace, which is enough for tool-call correctness, judging, turn-taking
and false-completion detection. They cannot give you transcription accuracy: a
hosted platform returns what its recognizer *heard* and never what was actually
*said*, so there is no reference to score against and `Turn.spoken` is left None.
Transcription metrics need audio you control — see `run_streaming`.

That limit is worth stating plainly, because it is the difference between
"evaluating your agent" and "evaluating your agent's ears". A platform trace can
tell you the agent confidently ordered the wrong dish. Only your own audio can
tell you it was because the recognizer misheard.

Shapes are taken from each vendor's published webhook documentation. They are
exercised against representative fixtures, **not against a live account**, so
treat a parsing failure as a bug report rather than as your payload being wrong.
"""

from __future__ import annotations

import json
from typing import Any

from .adapters import CallTrace
from .case import Turn
from .metrics.tools import ToolCall

__all__ = ["EmptyPayloadError", "from_openai_messages", "from_retell", "from_vapi"]


class EmptyPayloadError(ValueError):
    """No conversation could be recovered from a payload.

    Raised rather than returning an empty trace, because an empty trace scores
    as a clean conversation: no failed tool assertions, no transcription errors,
    nothing for the judge to object to. A dropped webhook would quietly read as
    a passing call, which is the worst possible way to be wrong about a payload.
    """


def _unwrap(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """The vendor's envelope if present, else the payload itself.

    Both Retell and Vapi nest the interesting fields one level down, but users
    routinely hand over the already-unwrapped inner object. Accepting either
    costs one function and removes a whole class of "why is my payload empty".
    """
    value = payload.get(key)
    return value if isinstance(value, dict) else payload


def _sub(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    """Tool arguments arrive parsed from some vendors and as a JSON string from
    others — occasionally from the same vendor across API versions."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def from_openai_messages(
    messages: list[dict[str, Any]],
    caller_role: str = "user",
    agent_role: str = "assistant",
) -> CallTrace:
    """Map OpenAI-format chat messages to a trace.

    The workhorse. LiveKit Agents, Pipecat, custom stacks and most vendors can
    emit this shape, so platform-specific adapters normalize into it rather than
    each reimplementing turn assembly.

    Tool calls attach to the turn whose caller utterance preceded them, which is
    what makes "the agent claimed it booked something on the turn it never called
    the tool" detectable per-turn rather than only per-call.
    """
    turns: list[Turn] = []
    pending_agent: list[str] = []
    pending_tools: list[ToolCall] = []

    def flush_into_last() -> None:
        if not turns:
            return
        if pending_agent:
            existing = turns[-1].agent_said
            joined = " ".join(pending_agent).strip()
            turns[-1].agent_said = f"{existing} {joined}".strip() if existing else joined
        turns[-1].tool_calls.extend(pending_tools)
        pending_agent.clear()
        pending_tools.clear()

    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()

        if role == caller_role:
            if not turns and (pending_agent or pending_tools):
                # The agent spoke before the caller did — which is how nearly
                # every real call opens. Without its own turn, the greeting is
                # flushed onto the FIRST caller turn and the transcript renders
                # as "Customer: one kung pao chicken / Agent: how can I help?",
                # reversing cause and effect. A judge reading that sees an agent
                # ignoring the order it was just given.
                turns.append(Turn(heard="", agent_said=" ".join(pending_agent).strip(),
                                  tool_calls=list(pending_tools)))
                pending_agent.clear()
                pending_tools.clear()
            else:
                flush_into_last()
            turns.append(Turn(heard=content))
            continue

        if role == agent_role:
            if content:
                pending_agent.append(content)
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                name = fn.get("name") or call.get("name")
                if name:
                    pending_tools.append(ToolCall(
                        name=name,
                        arguments=_as_dict(fn.get("arguments", call.get("arguments"))),
                    ))

    flush_into_last()
    if not turns:
        # An agent-only call (voicemail, immediate hangup) still has a
        # transcript worth judging; represent it as one turn with no caller text.
        agent_text = " ".join(
            (m.get("content") or "").strip()
            for m in messages
            if m.get("role") == agent_role and (m.get("content") or "").strip()
        ).strip()
        if agent_text or pending_tools:
            turns = [Turn(heard="", agent_said=agent_text, tool_calls=list(pending_tools))]

    if not turns:
        raise EmptyPayloadError(
            f"no conversation in {len(messages)} message(s): expected roles "
            f"{caller_role!r}/{agent_role!r}")
    if not any(t.received or t.agent_said or t.tool_calls for t in turns):
        # Turns present, all of them empty. Structurally a conversation,
        # semantically nothing — the signature of reading the wrong text field
        # (Vapi stores it under `message`, not `content`). Silently scoring this
        # reports a clean call in which nobody spoke.
        raise EmptyPayloadError(
            f"{len(turns)} turn(s) parsed but every one is empty — is the text "
            f"under a different key than 'content'?")
    return CallTrace(turns=turns)


def from_retell(payload: dict[str, Any]) -> CallTrace:
    """Map a Retell `call_ended` / `call_analyzed` webhook payload.

    Retell nests the interesting data under `call` and offers three views of the
    conversation. `transcript_with_tool_calls` is preferred because it is the
    only one that interleaves tool invocations with speech — `transcript_object`
    drops them silently, which would make every tool assertion pass vacuously.
    """
    call = _unwrap(payload, "call")

    entries = call.get("transcript_with_tool_calls") or call.get("transcript_object") or []
    if entries:
        return _from_retell_entries(entries)

    # Last resort: the flat `transcript` string. No tool calls in it, so tool
    # expectations on this trace would pass without evidence — return it as a
    # flat transcript so it is judged but not tool-checked.
    transcript = (call.get("transcript") or "").strip()
    if not transcript:
        raise EmptyPayloadError(
            "retell payload has no transcript_with_tool_calls, transcript_object "
            "or transcript")
    return CallTrace(transcript=transcript)


def _from_retell_entries(entries: list[dict[str, Any]]) -> CallTrace:
    turns: list[Turn] = []
    pending_agent: list[str] = []
    pending_tools: list[ToolCall] = []

    def flush() -> None:
        if not turns:
            return
        if pending_agent:
            turns[-1].agent_said = " ".join(pending_agent).strip()
        turns[-1].tool_calls.extend(pending_tools)
        pending_agent.clear()
        pending_tools.clear()

    for entry in entries:
        role = entry.get("role")
        content = (entry.get("content") or "").strip()

        if role == "user":
            flush()
            turns.append(Turn(heard=content))
        elif role == "agent":
            if content:
                pending_agent.append(content)
        elif role == "tool_call_invocation":
            name = entry.get("name")
            if name:
                pending_tools.append(
                    ToolCall(name=name, arguments=_as_dict(entry.get("arguments"))))
        # tool_call_result carries the response, not the invocation. Ignored:
        # counting it would double every tool call and trip duplicate-commit.

    flush()
    if not turns:
        raise EmptyPayloadError("retell transcript entries contained no user turns")
    return CallTrace(turns=turns)


def _normalize_vapi_message(entry: dict[str, Any]) -> dict[str, Any]:
    """Vapi's native `{"role", "message"}` -> OpenAI's `{"role", "content"}`.

    Tool calls are carried through under either spelling. `role: "system"` entries
    (Vapi includes the full system prompt in artifact.messages) pass through and
    are ignored downstream, which is correct — the system prompt is not something
    the caller said.
    """
    normalized = dict(entry)
    if "content" not in normalized and "message" in normalized:
        normalized["content"] = normalized.get("message")
    return normalized


def from_vapi(payload: dict[str, Any]) -> CallTrace:
    """Map a Vapi `end-of-call-report` webhook payload.

    Vapi wraps everything in `message` and keeps the conversation in
    `message.artifact.messages` (OpenAI-shaped). Tool calls appear inline there,
    and `message.toolCalls` may also carry them at the top level for tool-call
    events — both are read, and duplicates are dropped, because counting a call
    twice would trip the duplicate-commit check on a healthy agent.
    """
    message = _unwrap(payload, "message")
    artifact = _sub(message, "artifact")

    # `messagesOpenAIFormatted` is preferred because it already uses `content`.
    # `artifact.messages` is Vapi's native shape and stores the text under
    # `message`, so handing it straight to from_openai_messages produces a turn
    # per utterance with empty text — a conversation that is structurally present
    # and semantically gone. That slips past the empty-payload guard, because
    # turns exist; it just scores a call in which nobody said anything.
    messages = artifact.get("messagesOpenAIFormatted") or []
    if not messages:
        messages = [_normalize_vapi_message(m) for m in (artifact.get("messages") or [])]
    if messages:
        trace = from_openai_messages(messages)
    else:
        transcript = (message.get("transcript") or artifact.get("transcript") or "").strip()
        if not transcript and not message.get("toolCalls"):
            raise EmptyPayloadError(
                "vapi payload has no artifact.messages, transcript or toolCalls")
        trace = CallTrace(transcript=transcript or "(no transcript)")

    top_level = [
        ToolCall(name=_sub(c, "function").get("name") or c.get("name") or "",
                 arguments=_as_dict(_sub(c, "function").get("arguments")))
        for c in (message.get("toolCalls") or [])
    ]
    top_level = [c for c in top_level if c.name]
    if not top_level:
        return trace

    seen = {(c.name, json.dumps(c.arguments, sort_keys=True))
            for t in trace.turns for c in t.tool_calls}
    extra = [c for c in top_level
             if (c.name, json.dumps(c.arguments, sort_keys=True)) not in seen]
    if extra:
        if trace.turns:
            trace.turns[-1].tool_calls.extend(extra)
        else:
            trace = CallTrace(transcript=trace.transcript or "",
                              turns=[Turn(heard="", tool_calls=extra)])
    return trace
