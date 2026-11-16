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

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from .audio import AudioBuffer
from .case import ConversationCase, Turn, TurnCase
from .gate import Gate
from .metrics.judge import Judge
from .metrics.latency import TurnLatency
from .metrics.semantic import Similarity
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

    def __call__(self, audio: Path) -> str | Transcription: ...


@dataclass
class Transcription:
    """A richer transcriber result. Every field beyond `text` is optional, because
    providers differ wildly in what they expose and a missing field should degrade
    the report rather than crash the run."""

    text: str
    confidence: float | None = None
    latency: TurnLatency | None = None


@dataclass
class Final:
    """One final transcript, with when it arrived relative to stream start."""

    text: str
    at_ms: float
    confidence: float | None = None


@dataclass
class StreamResult:
    """What a streaming recognizer produced for one utterance.

    **Not every streaming recognizer emits a final.** ElevenLabs Scribe v2
    realtime emits only interim transcripts unless server VAD is enabled, because
    it does no endpointing — turn boundaries are the caller's problem. A harness
    that collects `FINAL_TRANSCRIPT` and nothing else records that recognizer as
    producing empty transcripts, which is how a perfectly working provider gets
    written up as broken. If your adapter sees no finals, fall back to the last
    interim and say which you used.

    `audio_ended_at_ms` is when the last real audio frame was pushed — not the
    trailing silence. Finalization latency is measured from there, because that is
    the moment the caller stopped talking and started waiting.
    """

    finals: list[Final] = field(default_factory=list)
    interims: list[Final] = field(default_factory=list)
    """Optional. Only consulted when nothing finalized."""
    audio_ended_at_ms: float = 0.0

    @property
    def spoken_finals(self) -> list[Final]:
        """Finals that carried words. Recognizers commonly emit blank finals — a
        flush at endpoint, a turn of silence — and every property below ignores
        them: a blank final transcribed nothing, so it neither ends the wait nor
        has a confidence worth averaging."""
        return [f for f in self.finals if f.text.strip()]

    @property
    def text(self) -> str:
        spoken = self.spoken_finals
        if spoken:
            return " ".join(f.text.strip() for f in spoken)
        return self.interims[-1].text.strip() if self.interims else ""

    @property
    def used_interim_fallback(self) -> bool:
        """True when the transcript came from an interim because nothing
        finalized. Worth surfacing in a report: the text is usable, the timing
        is not."""
        return not self.spoken_finals and bool(self.interims)

    @property
    def finalization_latency_ms(self) -> float | None:
        """End of speech → last final. The number a batch API cannot give you.

        None when nothing was transcribed — an empty result is a transcription
        failure, not a latency of zero, and reporting 0ms would make a broken
        recognizer look like the fastest one in the comparison.
        """
        spoken = self.spoken_finals
        if not spoken:
            return None
        return max(0.0, max(f.at_ms for f in spoken) - self.audio_ended_at_ms)

    @property
    def mean_confidence(self) -> float | None:
        scored = [f.confidence for f in self.spoken_finals if f.confidence is not None]
        return sum(scored) / len(scored) if scored else None


@runtime_checkable
class StreamingTranscriber(Protocol):
    """A recognizer fed audio at real time, reporting finals with timestamps.

    Use this over `Transcriber` whenever you care about latency or endpointing.
    A batch interface cannot measure either: it returns text with no timeline, and
    it never exercises the turn-taking model that decides when a caller has
    stopped speaking.
    """

    def __call__(self, audio: AudioBuffer) -> StreamResult: ...


@dataclass
class CallTrace:
    """What one evaluated conversation produced.

    This is the contract an `Agent` returns. It mirrors what a voice stack already
    logs, so implementing it is usually reshaping existing data rather than
    instrumenting anything new.

    Prefer `turns`. Returning structured turns gets the transcript, tool calls and
    latencies derived for you, and — the part that matters — scores each turn's
    transcription alongside the call's behaviour. Returning only a flat
    `transcript` still works, but throws away the per-turn hearing metrics,
    because a joined string no longer says which words the recognizer got wrong.
    """

    transcript: str | None = None
    turns: list[Turn] = field(default_factory=list)
    tool_calls: list[ToolCall] | None = None
    """None derives from turns; an explicit `[]` asserts no tool fired."""
    utterances: list[Utterance] = field(default_factory=list)
    latencies: list[TurnLatency] | None = None

    def __post_init__(self) -> None:
        if not self.transcript and not self.turns:
            raise ValueError("CallTrace needs either turns or a transcript")


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
    similarity: Similarity | None = None


@dataclass
class Scenario:
    """A scripted conversation plus what the agent should have done."""

    id: str
    caller_turns: Sequence[str]
    expected_tools: ToolExpectation | None = None
    description: str = ""
    """What the caller wants, in one line. Passed to the judge as context."""
    expected_behavior: str = ""
    """The rubric for this scenario. Without it the scenario is not judged at
    all — see `ConversationCase.expected_behavior` for why a shared generic
    rubric is worse than none."""


def run_audio(cases: Sequence[AudioCase], transcriber: Transcriber) -> SuiteReport:
    """Transcribe each fixture and score the result.

        report = run_audio(cases, lambda p: my_stt.transcribe(p).text)
        sys.exit(report.gate().exit_code())

    For latency and endpointing, use `run_streaming` instead — a batch
    transcriber cannot report either.
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
            similarity=case.similarity,
        ))
    return suite.run()


def run_streaming(
    cases: Sequence[AudioCase],
    transcriber: StreamingTranscriber,
    noise: AudioBuffer | None = None,
    snr_db: float = 6.0,
) -> SuiteReport:
    """Stream each fixture through a live recognizer and score text *and* latency.

        report = run_streaming(cases, my_streaming_stt)
        report = run_streaming(cases, my_streaming_stt, noise=cafe, snr_db=6)

    Passing `noise` mixes it into every fixture at `snr_db` before streaming,
    which is how you test the failure that actually loses orders: a recognizer
    that is perfect in a quiet room and drops the product name over a busy one.

    Finalization latency lands in the report as the STT stage, so the standard
    latency thresholds apply without extra wiring.
    """
    from .audio import load_wav, mix_at_snr

    suite = Suite("streaming-audio")
    for case in cases:
        buffer = load_wav(case.audio)
        if noise is not None:
            buffer = mix_at_snr(buffer, noise, snr_db)

        result = transcriber(buffer)
        latency = (
            TurnLatency(vad_to_stt_ms=result.finalization_latency_ms)
            if result.finalization_latency_ms is not None
            else None
        )
        suite.add_turn(TurnCase(
            id=case.id,
            hypothesis=result.text,
            reference=case.reference,
            terms=case.terms,
            languages=case.languages,
            latency=latency,
            confidence=result.mean_confidence,
            score_wer=case.score_wer,
            similarity=case.similarity,
        ))
    return suite.run()


def run_scenarios(
    scenarios: Sequence[Scenario],
    agent: Agent,
    judge: Judge | None = None,
) -> SuiteReport:
    """Run each scripted conversation against the agent and score the traces.

        report = run_scenarios(scenarios, my_agent_replay)
        sys.exit(report.gate().exit_code())

    Pass `judge` to add model-based quality scoring on top of the deterministic
    checks. Only scenarios that define `expected_behavior` are judged, so you can
    add a rubric to the handful of cases where quality is the question and leave
    the rest checked by tool trace alone — which is cheaper and more reliable.
    """
    suite = Suite("scenarios")
    for scenario in scenarios:
        trace = agent(scenario.caller_turns)
        suite.add_conversation(ConversationCase(
            id=scenario.id,
            transcript=trace.transcript,
            turns=trace.turns,
            tool_calls=trace.tool_calls,
            expected_tools=scenario.expected_tools,
            utterances=trace.utterances,
            latencies=trace.latencies,
            judge=judge,
            scenario=scenario.description,
            expected_behavior=scenario.expected_behavior,
        ))
    return suite.run()


def assert_gate(report: SuiteReport, gate: Gate | None = None) -> None:
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
