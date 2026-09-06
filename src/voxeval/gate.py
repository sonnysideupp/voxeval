"""Thresholds — turn measurements into a pass/fail a deploy script can use.

A metric nobody blocks on is a dashboard. The point of this library is to sit in
CI or a pre-deploy script and refuse to ship a regression, which requires taking
a position on what "bad enough to stop" means.

Two design choices worth stating, because both were learned the expensive way:

**Every threshold carries a margin.** A check written as `new <= old` looks
rigorous and is unusable: when the baseline is already perfect, it demands
perfection forever, and a single-token difference — well inside the measurement's
own noise — aborts a deploy. A gate that fails one run in three gets ignored, and
an ignored gate protects nothing.

**Some checks have no margin.** A commit tool firing twice or an agent claiming
"all set" without committing are not noisy measurements; they are defects. Those
are booleans, and they fail at one occurrence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """WARN records a problem and lets the deploy proceed; FAIL blocks it.

    Use WARN while calibrating a new check against real traffic. Promote it to
    FAIL once you know its false-positive rate — shipping a check straight to
    FAIL is how gates get disabled.
    """

    WARN = "warn"
    FAIL = "fail"


@dataclass
class Threshold:
    """One named check.

    Exactly one of `max_value` / `min_value` should be set. `margin` widens the
    bound in the permissive direction and defaults to zero, so a caller who wants
    tolerance has to say so — silent slack is worse than none.
    """

    name: str
    max_value: float | None = None
    min_value: float | None = None
    margin: float = 0.0
    severity: Severity = Severity.FAIL
    note: str = ""

    def __post_init__(self) -> None:
        if (self.max_value is None) == (self.min_value is None):
            raise ValueError(f"{self.name}: set exactly one of max_value or min_value")
        if self.margin < 0:
            raise ValueError(f"{self.name}: margin must be >= 0")

    def evaluate(self, value: float | None) -> "ThresholdResult":
        """Missing values pass.

        Deliberate: a pipeline that does not expose a stage (a speech-to-speech
        model has no transcript latency) should not fail a check it cannot answer.
        Absence of evidence is not a regression — if you need a metric to exist,
        assert that separately.
        """
        if value is None:
            return ThresholdResult(self, None, passed=True, skipped=True)

        if self.max_value is not None:
            bound = self.max_value + self.margin
            passed = value <= bound
        else:
            bound = self.min_value - self.margin
            passed = value >= bound
        return ThresholdResult(self, value, passed=passed, bound=bound)


@dataclass
class ThresholdResult:
    threshold: Threshold
    value: float | None
    passed: bool
    bound: float | None = None
    skipped: bool = False

    def __str__(self) -> str:
        if self.skipped:
            return f"  ○  {self.threshold.name}: not measured"
        mark = "✓" if self.passed else ("!" if self.threshold.severity is Severity.WARN else "✗")
        direction = "<=" if self.threshold.max_value is not None else ">="
        line = f"  {mark}  {self.threshold.name}: {self.value:.4g} {direction} {self.bound:.4g}"
        if not self.passed and self.threshold.note:
            line += f"\n       {self.threshold.note}"
        return line


@dataclass
class GateResult:
    results: list[ThresholdResult] = field(default_factory=list)

    @property
    def failures(self) -> list[ThresholdResult]:
        return [
            r for r in self.results
            if not r.passed and r.threshold.severity is Severity.FAIL
        ]

    @property
    def warnings(self) -> list[ThresholdResult]:
        return [
            r for r in self.results
            if not r.passed and r.threshold.severity is Severity.WARN
        ]

    @property
    def passed(self) -> bool:
        return not self.failures

    def exit_code(self) -> int:
        """0 to proceed, 1 to block. For `sys.exit(gate.exit_code())`."""
        return 0 if self.passed else 1

    def __str__(self) -> str:
        lines = [str(r) for r in self.results]
        lines.append("")
        if self.passed:
            suffix = f" ({len(self.warnings)} warning(s))" if self.warnings else ""
            lines.append(f"PASS — {len(self.results)} check(s){suffix}")
        else:
            lines.append(f"FAIL — {len(self.failures)} of {len(self.results)} check(s) blocked the deploy")
        return "\n".join(lines)


class Gate:
    """A named set of thresholds evaluated against a metrics dict.

        gate = Gate("pre-deploy")
        gate.add(Threshold("wer", max_value=0.15, margin=0.05))
        gate.add(Threshold("term_recall", min_value=0.90))
        result = gate.check({"wer": 0.12, "term_recall": 1.0})
        sys.exit(result.exit_code())
    """

    def __init__(self, name: str = "gate") -> None:
        self.name = name
        self._thresholds: list[Threshold] = []

    def add(self, threshold: Threshold) -> "Gate":
        self._thresholds.append(threshold)
        return self

    def check(self, metrics: dict[str, float | None]) -> GateResult:
        return GateResult([t.evaluate(metrics.get(t.name)) for t in self._thresholds])


def default_gate() -> Gate:
    """A starting set of thresholds for a phone-based voice agent.

    These are opinionated defaults, not universal truths — calibrate them against
    your own traffic before trusting them. The latency numbers reflect commonly
    cited targets for phone agents (first audio inside about a second, and a tail
    that stays under three seconds); the accuracy numbers assume a domain where
    specific nouns matter more than overall wording.
    """
    gate = Gate("voice-agent pre-deploy")
    gate.add(Threshold(
        "term_recall", min_value=0.90, margin=0.05,
        note="A domain term was lost. Check recognizer keyterm/context biasing.",
    ))
    gate.add(Threshold(
        "wer", max_value=0.20, margin=0.06,
        note="Margin is ~1 token on a short utterance; a real regression moves more.",
    ))
    gate.add(Threshold(
        "collapses", max_value=0, severity=Severity.FAIL,
        note="A known mis-hear reappeared — the recognizer produced a plausible "
             "wrong answer, which is worse than an obviously wrong one.",
    ))
    gate.add(Threshold(
        "out_of_set_language_rate", max_value=0.0,
        note="Recognizer returned a language the agent does not support. It will "
             "act on a sentence nobody said.",
    ))
    gate.add(Threshold("end_to_end_p50_ms", max_value=1200.0, margin=200.0))
    gate.add(Threshold("end_to_end_p95_ms", max_value=3000.0, margin=500.0))
    gate.add(Threshold(
        "barge_in_recovery_p95_ms", max_value=500.0, margin=200.0,
        severity=Severity.WARN,
        note="Agent kept talking after being interrupted; reads as not listening.",
    ))
    gate.add(Threshold(
        "tool_call_failures", max_value=0,
        note="A required tool was never called, a forbidden one was, or arguments "
             "did not match. The transcript may still read as a successful call.",
    ))
    gate.add(Threshold(
        "duplicate_commits", max_value=0,
        note="A commit tool fired more than once. This double-charges customers.",
    ))
    gate.add(Threshold(
        "claimed_without_commit", max_value=0,
        note="Agent said the task was done without calling the tool that does it. "
             "The caller believes they have a booking that does not exist.",
    ))
    return gate
