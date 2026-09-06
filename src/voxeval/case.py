"""Cases and results — the unit of evaluation.

A `TurnCase` scores one utterance: what was said, what the recognizer heard, how
long each stage took. A `ConversationCase` scores a whole call: the transcript,
the tool trace, and the timeline.

Both are plain dataclasses you construct from whatever your stack already logs.
This library deliberately does not run your agent, own your audio pipeline, or
require a provider SDK — it scores traces. That keeps it usable against a live
call recording, a replay harness, or a synthetic fixture without changing the
metric code, and it means adding a provider never means changing this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .metrics.judge import Judge, Verdict, safe_judge
from .metrics.language import LanguageCheck, check_language
from .metrics.latency import TurnLatency
from .metrics.semantic import Similarity, semantic_wer
from .metrics.tools import (
    ToolCall,
    ToolCallResult,
    ToolExpectation,
    check_tool_calls,
    claimed_without_committing,
)
from .metrics.transcription import (
    Term,
    TermRecallResult,
    digit_accuracy,
    extract_digits,
    term_recall,
    word_error_rate,
)
from .metrics.turntaking import TurnTakingReport, Utterance, analyze_turns


@dataclass
class TurnCase:
    """One utterance: reference text, what the recognizer returned, timings.

    `reference` is what the speaker actually said. For synthetic fixtures that is
    the TTS input; for real call audio it is a human transcription. Term recall
    works without a reference at all, which matters because real recordings are
    expensive to transcribe and the terms are usually the thing you care about.
    """

    id: str
    hypothesis: str
    reference: str | None = None
    terms: list[Term] = field(default_factory=list)
    latency: TurnLatency | None = None
    languages: list[str] | None = None
    confidence: float | None = None
    similarity: Similarity | None = None
    """Optional meaning-equivalence scorer for genuine paraphrase.

    Leave unset for recognizer evaluation: a transcriber is supposed to reproduce
    the words, so `semantic_wer` already answers the question and costs nothing.
    Set it when the "reference" is an intent rather than a script — an agent that
    may legitimately answer "we close at nine" or "nine PM, last order eight
    thirty" cannot be scored by edit distance against either one."""
    score_wer: bool = True
    """Set False when WER is a known-invalid metric for this utterance.

    The case that forces this: a spoken phone number. "six one seven" against
    "617" is a WER of 1.0 and the two are the same number. Including such a turn
    in the WER aggregate doesn't just misreport it — it inflates the suite-wide
    average toward the threshold and eventually fails a healthy deploy for a turn
    that was transcribed perfectly.

    Digit accuracy still scores; only WER is suppressed."""

    def evaluate(self) -> TurnResult:
        return TurnResult(
            case=self,
            wer=(
                word_error_rate(self.reference, self.hypothesis)
                if self.reference and self.score_wer
                else None
            ),
            # Always computed alongside raw WER when a reference exists: it is
            # deterministic and free, and the pair is what makes either one
            # readable. Raw WER alone cannot distinguish "the recognizer wrote
            # 6:30 instead of six thirty" from "the recognizer lost the time".
            semantic_wer=(
                semantic_wer(self.reference, self.hypothesis)
                if self.reference and self.score_wer
                else None
            ),
            similarity=(
                self.similarity(self.reference, self.hypothesis)
                if self.similarity is not None and self.reference
                else None
            ),
            terms=term_recall(self.hypothesis, self.terms) if self.terms else None,
            # Decide on EXTRACTED digits, not literal digit characters. A spoken
            # number — "six one seven" — contains no digit chars, and that is
            # precisely the case this metric exists for. Checking `isdigit()`
            # skipped it silently.
            digits=(
                digit_accuracy(self.reference, self.hypothesis)
                if self.reference and extract_digits(self.reference)
                else None
            ),
            language=check_language(self.hypothesis, self.languages) if self.languages else None,
        )


@dataclass
class TurnResult:
    case: TurnCase
    wer: float | None = None
    semantic_wer: float | None = None
    similarity: float | None = None
    terms: TermRecallResult | None = None
    digits: float | None = None
    language: LanguageCheck | None = None

    @property
    def ok(self) -> bool:
        """A turn fails on a collapse or an out-of-set language regardless of WER —
        both mean the agent received a confident wrong answer rather than a
        visibly degraded one."""
        if self.terms and self.terms.has_collapse:
            return False
        return not (self.language and not self.language.ok)

    def __str__(self) -> str:
        bits = [f"{self.case.id}"]
        if self.wer is not None:
            bits.append(f"wer={self.wer:.0%}")
        if self.semantic_wer is not None and self.semantic_wer != self.wer:
            # Only shown when it differs — printing an identical number twice
            # trains the reader to skip the column that matters.
            bits.append(f"sem_wer={self.semantic_wer:.0%}")
        if self.similarity is not None:
            bits.append(f"sim={self.similarity:.2f}")
        if self.terms is not None:
            total = len(self.terms.found) + len(self.terms.missed)
            bits.append(f"terms={len(self.terms.found)}/{total}")
            if self.terms.collapsed:
                bits.append(f"COLLAPSE={self.terms.collapsed}")
        if self.digits is not None:
            bits.append(f"digits={self.digits:.0%}")
        if self.language and not self.language.ok:
            bits.append(str(self.language))
        return "  ".join(bits)



@dataclass
class Turn:
    """One exchange in a conversation: what the caller said, what the agent heard,
    what it said back, and what it did.

    This is the abstraction that makes voice and text the same object rather than
    two parallel ones. The only structural difference between a voice agent and a
    text agent is that voice has a gap between `spoken` and `heard`:

        voice   spoken="kung pao chicken"  heard="compound chicken"   <- gap
        text    spoken="kung pao chicken"  heard=None                 <- no gap

    Everything downstream — tool traces, latency, judging, turn-taking — is
    identical for both. Only the transcription metrics switch on, and they switch
    on exactly when there is something to compare.

    That gap is the whole reason this library exists. A text harness cannot
    represent it, so it cannot catch the failure where a slightly-wrong transcript
    makes a perfectly-behaved agent do the wrong thing.
    """

    spoken: str | None = None
    """Ground truth: the TTS input, or a human transcription of real call audio.
    None when unknown — term recall still works without it."""

    heard: str | None = None
    """What the recognizer returned. None for a text-mode agent, which hears
    exactly what was said."""

    agent_said: str = ""
    """The agent's spoken reply. Judged; never scored for transcription."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    latency: TurnLatency | None = None
    terms: list[Term] = field(default_factory=list)
    languages: list[str] | None = None
    confidence: float | None = None
    score_wer: bool = True

    @property
    def received(self) -> str:
        """What the agent actually acted on.

        `heard` when a recognizer was involved, `spoken` otherwise. Judging and
        tool-trace analysis use this, because holding an agent to words it never
        received is scoring the recognizer twice.
        """
        return self.heard if self.heard is not None else (self.spoken or "")

    @property
    def is_voice(self) -> bool:
        """True when there is a transcription gap to measure."""
        return self.spoken is not None and self.heard is not None

    def to_turn_case(self, case_id: str) -> TurnCase:
        return TurnCase(
            id=case_id,
            hypothesis=self.received,
            reference=self.spoken if self.is_voice else None,
            terms=self.terms,
            latency=self.latency,
            languages=self.languages,
            confidence=self.confidence,
            score_wer=self.score_wer,
        )


@dataclass
class ConversationCase:
    """A whole call: transcript, tool trace, and optionally the turn timeline.

    Two ways to build one, and `turns` is the better one:

        # Voice or text, structured — everything derives from the turns
        ConversationCase(id="order", turns=[
            Turn(spoken="one kung pao chicken", heard="one compound chicken",
                 agent_said="Sorry, we don't have that.", terms=[...]),
        ])

        # Flat, when all you have is a logged transcript string
        ConversationCase(id="order", transcript="Customer: ...\nAgent: ...")

    With `turns`, the transcript, tool calls and latencies are derived, and each
    turn is *also* scored for transcription — so one object produces both the
    per-turn hearing metrics and the whole-call behaviour metrics instead of you
    hand-maintaining two parallel lists linked by an id convention.
    """

    id: str
    transcript: str | None = None
    """None means "derive from turns". An explicitly supplied string wins."""
    turns: list[Turn] = field(default_factory=list)
    tool_calls: list[ToolCall] | None = None
    """None means "derive from turns". An explicit `[]` is an assertion that NO
    tool fired, and is preserved — see __post_init__."""
    expected_tools: ToolExpectation | None = None
    utterances: list[Utterance] = field(default_factory=list)
    latencies: list[TurnLatency] | None = None

    judge: Judge | None = None
    """Optional model-based scorer for conversation quality."""
    scenario: str = ""
    """What the caller was trying to do, in one line. Given to the judge."""
    expected_behavior: str = ""
    """What a good agent does here, in plain language — the rubric.

    Write it as observable behaviour ("quotes a price only if it appears in the
    menu context"), not as a vibe ("is helpful"). A vague rubric is the main
    reason LLM judges score the same transcript 85 one run and 61 the next, and
    that variance lands on your deploy gate."""

    def __post_init__(self) -> None:
        # Derive ONLY what was omitted (None), never what was supplied.
        #
        # The distinction matters because `[]` is a claim, not an absence. A
        # caller passing tool_calls=[] is asserting no tool fired; if that were
        # treated as "unset" and refilled from the turns, a `required` check
        # could pass on a trace the caller explicitly said was empty — the
        # library silently contradicting its input on the one metric where
        # being wrong means a customer was charged.
        if self.transcript is None:
            self.transcript = self.render_transcript() if self.turns else ""
        if self.tool_calls is None:
            self.tool_calls = [c for t in self.turns for c in t.tool_calls]
        if self.latencies is None:
            self.latencies = [t.latency for t in self.turns if t.latency is not None]

    def render_transcript(self) -> str:
        """The spoken conversation, as the judge should see it.

        Built from `received`, not `spoken` — the agent replied to what it heard,
        and judging it against words it never got is scoring the recognizer twice
        under a different name. Transcription error is measured separately, per
        turn, where it can be attributed.
        """
        lines: list[str] = []
        for turn in self.turns:
            if turn.received:
                lines.append(f"Customer: {turn.received}")
            if turn.agent_said:
                lines.append(f"Agent: {turn.agent_said}")
        return "\n".join(lines)

    def turn_cases(self) -> list[TurnCase]:
        return [t.to_turn_case(f"{self.id}#{i}") for i, t in enumerate(self.turns)]

    def evaluate(self) -> ConversationResult:
        commit_tools = self.expected_tools.commit_tools if self.expected_tools else ()
        # __post_init__ has filled these; the locals exist so the annotations
        # stay honest about what a caller may pass.
        transcript = self.transcript or ""
        tool_calls = self.tool_calls if self.tool_calls is not None else []
        return ConversationResult(
            case=self,
            turns=[c.evaluate() for c in self.turn_cases()],
            # Judged only when a rubric exists. Asking a model "was this good?"
            # with nothing to compare against produces a number that drifts with
            # the model version and measures nothing.
            verdict=(
                safe_judge(self.judge, self.scenario, self.expected_behavior, transcript)
                if self.judge is not None and self.expected_behavior
                else None
            ),
            tools=(
                check_tool_calls(tool_calls, self.expected_tools)
                if self.expected_tools
                else None
            ),
            turn_taking=analyze_turns(self.utterances) if self.utterances else None,
            false_completion=(
                claimed_without_committing(transcript, tool_calls, commit_tools)
                if commit_tools else False
            ),
        )


@dataclass
class ConversationResult:
    case: ConversationCase
    turns: list[TurnResult] = field(default_factory=list)
    tools: ToolCallResult | None = None
    turn_taking: TurnTakingReport | None = None
    false_completion: bool = False
    verdict: Verdict | None = None

    @property
    def ok(self) -> bool:
        """Deliberately ignores the judge.

        A conversation is not "failed" because a model scored it 68. The judge
        feeds an aggregate that the gate treats as a WARN; letting it flip a
        per-case boolean would make individual results non-reproducible."""
        if self.false_completion:
            return False
        if any(not t.ok for t in self.turns):
            return False
        return not (self.tools and not self.tools.ok)

    def __str__(self) -> str:
        bits = [self.case.id]
        if self.turns:
            wers = [t.wer for t in self.turns if t.wer is not None]
            if wers:
                bits.append(f"{len(self.turns)} turns wer={sum(wers) / len(wers):.0%}")
            else:
                bits.append(f"{len(self.turns)} turns")
        if self.tools:
            bits.append(str(self.tools))
        if self.false_completion:
            bits.append("CLAIMED COMPLETION WITHOUT COMMITTING")
        if self.turn_taking:
            bits.append(str(self.turn_taking))
        if self.verdict is not None:
            bits.append(str(self.verdict))
        return "  ".join(bits)
