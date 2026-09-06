"""Point voxeval at your agent instead of hand-building traces.

The core library scores traces, which is the honest primitive: a voice agent
cannot be invoked as `agent("hello")` the way a text model can, because it needs
audio, a media stream, and usually a phone leg. There is no equivalent of
`openai:gpt-4` you can name in a config file, and no cross-vendor trace format
that voice stacks agree on — the closest thing is the OpenTelemetry GenAI
conventions, which say nothing about barge-in or STT finalization.

But making every user write glue is a bad default. So this module defines the two
callables that actually matter, and runs suites against them:

    Transcriber   audio  →  text          (evaluating the recognizer)
    Agent         caller turns → trace     (evaluating the whole conversation)

Implement whichever you need. Everything else — scoring, aggregation, the gate —
is unchanged, so an adapter is the only code you own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence, runtime_checkable

from .case import ConversationCase, TurnCase
from .metrics.latency import TurnLatency
from .metrics.tools import ToolCall, ToolExpectation
from .metrics.transcription import Term
from .metrics.turntaking import Utterance
from .suite import Suite, SuiteReport


@runtime_checkable
class Transcriber(Protocol):
    """Anything that turns audio into text.

    Deliberately the narrowest possible interface — a bare function satisfies it,
    so wrapping Deepgram, Whisper, Soniox or a local model is a lambda rather than
    a subclass. Return either the transcript or a `Transcription` when you can
    also report confidence and timing.
    """

    def __call__(self, audio: Path) -> "str | Transcription": ...


@dataclass
class Transcription:
    """A richer transcriber result. Every field beyond `text` is optional, because
    providers differ wildly in what they expose and a missing field should degrade
    the report rather than crash the run."""

    text: str
    confidence: float | None = None
    latency: TurnLatency | None = None


@dataclass
class CallTrace:
    """What one evaluated conversation produced.

    This is the contract an `Agent` returns. It mirrors what a voice stack already
    logs, so implementing it is usually reshaping existing data rather than
    instrumenting anything new.
    """

    transcript: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    utterances: list[Utterance] = field(default_factory=list)
    latencies: list[TurnLatency] = field(default_factory=list)


@runtime_checkable
class Agent(Protocol):
    """Anything that can hold a scripted conversation and report what happened.

    `caller_turns` are the things the caller says, in order. Most teams implement
    this against a text-mode replay of their agent rather than real audio: it runs
    in CI in seconds, needs no media stack, and still exercises the prompt, the
    tools, and the business logic — which is where regressions actually live.

    Evaluate the audio path separately with a `Transcriber`. Splitting the two is
    what keeps this runnable on every commit instead of nightly.
    """

    def __call__(self, caller_turns: Sequence[str]) -> CallTrace: ...


@dataclass
class AudioCase:
    """An audio fixture plus what should come back out of it."""

    id: str
    audio: Path
    reference: str | None = None
    terms: list[Term] = field(default_factory=list)
    languages: list[str] | None = None
    score_wer: bool = True


@dataclass
class Scenario:
    """A scripted conversation plus what the agent should have done."""

    id: str
    caller_turns: Sequence[str]
    expected_tools: ToolExpectation | None = None


def run_audio(cases: Sequence[AudioCase], transcriber: Transcriber) -> SuiteReport:
    """Transcribe each fixture and score the result.

        report = run_audio(cases, lambda p: my_stt.transcribe(p).text)
        sys.exit(report.gate().exit_code())
    """
    suite = Suite("audio")
    for case in cases:
        result = transcriber(case.audio)
        transcription = result if isinstance(result, Transcription) else Transcription(text=result)
        suite.add_turn(TurnCase(
            id=case.id,
            hypothesis=transcription.text,
            reference=case.reference,
            terms=case.terms,
            languages=case.languages,
            latency=transcription.latency,
            confidence=transcription.confidence,
            score_wer=case.score_wer,
        ))
    return suite.run()


def run_scenarios(scenarios: Sequence[Scenario], agent: Agent) -> SuiteReport:
    """Run each scripted conversation against the agent and score the traces.

        report = run_scenarios(scenarios, my_agent_replay)
        sys.exit(report.gate().exit_code())
    """
    suite = Suite("scenarios")
    for scenario in scenarios:
        trace = agent(scenario.caller_turns)
        suite.add_conversation(ConversationCase(
            id=scenario.id,
            transcript=trace.transcript,
            tool_calls=trace.tool_calls,
            expected_tools=scenario.expected_tools,
            utterances=trace.utterances,
            latencies=trace.latencies,
        ))
    return suite.run()


def assert_gate(report: SuiteReport, gate=None) -> None:
    """Raise `AssertionError` with the full gate output if any check failed.

    For use inside pytest, so voice regressions fail the same way everything else
    in a test suite does:

        def test_voice_agent_has_not_regressed():
            assert_gate(run_scenarios(SCENARIOS, my_agent))

    The message carries the whole report rather than just the first failure —
    when a gate trips you almost always want to know whether it was one metric or
    everything at once.
    """
    result = report.gate(gate)
    if not result.passed:
        raise AssertionError(f"voice agent gate failed\n\n{report}\n\n{result}")


# A plain function is a valid Transcriber/Agent; these exist for readability at
# call sites and for `isinstance` checks in user code.
TranscriberFn = Callable[[Path], "str | Transcription"]
AgentFn = Callable[[Sequence[str]], CallTrace]
