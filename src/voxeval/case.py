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

from .metrics.language import LanguageCheck, check_language
from .metrics.latency import TurnLatency
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
class ConversationCase:
    """A whole call: transcript, tool trace, and optionally the turn timeline."""

    id: str
    transcript: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    expected_tools: ToolExpectation | None = None
    utterances: list[Utterance] = field(default_factory=list)
    latencies: list[TurnLatency] = field(default_factory=list)

    def evaluate(self) -> ConversationResult:
        commit_tools = self.expected_tools.commit_tools if self.expected_tools else ()
        return ConversationResult(
            case=self,
            tools=(
                check_tool_calls(self.tool_calls, self.expected_tools)
                if self.expected_tools
                else None
            ),
            turn_taking=analyze_turns(self.utterances) if self.utterances else None,
            false_completion=(
                claimed_without_committing(self.transcript, self.tool_calls, commit_tools)
                if commit_tools else False
            ),
        )


@dataclass
class ConversationResult:
    case: ConversationCase
    tools: ToolCallResult | None = None
    turn_taking: TurnTakingReport | None = None
    false_completion: bool = False

    @property
    def ok(self) -> bool:
        if self.false_completion:
            return False
        return not (self.tools and not self.tools.ok)

    def __str__(self) -> str:
        bits = [self.case.id]
        if self.tools:
            bits.append(str(self.tools))
        if self.false_completion:
            bits.append("CLAIMED COMPLETION WITHOUT COMMITTING")
        if self.turn_taking:
            bits.append(str(self.turn_taking))
        return "  ".join(bits)
