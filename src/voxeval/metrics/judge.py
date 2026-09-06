"""Conversation quality — was the answer *good*, not just structurally correct?

Everything else in this library is deterministic. Tool traces, latency, term
recall and script detection are checkable without a model, which is what makes
them safe to gate a deploy on.

Judgement is different in two ways, and it is worth being precise about which,
because the usual framing is wrong:

* **It costs money and time.** Every case is an API call.
* **It is only as reproducible as your sampling.** At temperature 0 a judge is
  effectively deterministic — the same transcript scored 65/100 eight times out
  of eight in testing, byte-identical reasoning each time. Raise the temperature
  and that goes away.

What is *not* reproducible is the thing being judged. In the same testing, one
scenario scored 40, 65 and 100 across three end-to-end runs — and none of that
spread was the judge. The agent sampled a materially different conversation each
time and the judge scored each one consistently. If your judge scores swing, look
at the agent before blaming the judge; a swinging score usually means the caller
is getting a different experience every call, which is a production finding, not
an eval artifact.

So judging is opt-in, isolated behind a callable like every other integration
here, and defaults to WARN. Not because the judge is random, but because it sits
downstream of everything that is: agent sampling, prompt drift, and the rubric
you wrote. Promote it to FAIL once you have measured the end-to-end spread on
your own scenarios.

What the judge is *for*: the questions no assertion can answer. Did the agent
address what the caller asked? Did it invent a price? Did it stay in the
caller's language? Was the refusal polite or curt?

What it is not for: anything checkable. Never ask a judge whether the order tool
fired — read the trace. This is not a style preference. A rubric here once said
the agent should "call placeTakeoutOrder exactly once", while the judge was
correctly shown only the spoken transcript; it inferred from the wording that the
tool had not fired and scored a perfectly correct call 40/100. The judge cannot
see what you did not show it, and it will guess rather than abstain.

Write rubrics as observable behaviour, and watch for phrasing that forbids
correct actions. "Quotes prices only from the menu it was given" sounds precise
and cost an agent 35 points for correctly adding two menu prices into a total.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Protocol, runtime_checkable


@dataclass
class Verdict:
    """One judged conversation.

    `score` is 0-100. `passed` defaults to score >= 70, a common threshold, but a
    judge can override it — some rubrics have hard disqualifiers that should fail
    at any score, like inventing a price.
    """

    score: float
    reasoning: str = ""
    passed: bool | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.passed is None:
            self.passed = self.score >= 70.0

    @property
    def errored(self) -> bool:
        return self.error is not None

    def __str__(self) -> str:
        if self.errored:
            return f"judge unavailable ({self.error})"
        mark = "pass" if self.passed else "FAIL"
        text = f"{self.score:.0f}/100 {mark}"
        return f"{text} — {self.reasoning[:160]}" if self.reasoning else text


@runtime_checkable
class Judge(Protocol):
    """Scores a conversation against what it was supposed to do.

    A bare function satisfies this, so wrapping any provider is a lambda:

        def judge(scenario, expected, transcript):
            reply = client.messages.create(...)
            return Verdict(score=..., reasoning=...)

    Returning a plain float is also accepted; it is wrapped as a `Verdict`.
    """

    def __call__(self, scenario: str, expected: str, transcript: str) -> Verdict | float: ...


def safe_judge(judge: Judge, scenario: str, expected: str, transcript: str) -> Verdict:
    """Run a judge, converting any failure into an errored verdict.

    A judge raising must never take down an eval run. The provider being down
    says nothing about the agent, and a crashed suite reports no results at all —
    which is strictly worse than reporting every deterministic metric plus one
    missing score.
    """
    try:
        result = judge(scenario, expected, transcript)
    except Exception as exc:  # noqa: BLE001 — deliberately broad; see docstring
        return Verdict(score=0.0, error=f"{type(exc).__name__}: {exc}", passed=True)

    if isinstance(result, Verdict):
        return result
    return Verdict(score=float(result))


@dataclass
class JudgeReport:
    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def scored(self) -> list[Verdict]:
        """Verdicts that actually produced a score. Errored ones are excluded from
        every aggregate — averaging a failed API call in as a zero would make a
        provider outage look like an agent regression."""
        return [v for v in self.verdicts if not v.errored]

    @property
    def mean_score(self) -> float | None:
        return mean(v.score for v in self.scored) if self.scored else None

    @property
    def pass_rate(self) -> float | None:
        if not self.scored:
            return None
        return sum(1 for v in self.scored if v.passed) / len(self.scored)

    @property
    def errors(self) -> int:
        return sum(1 for v in self.verdicts if v.errored)

    def __str__(self) -> str:
        if not self.scored:
            return f"judge: no scores ({self.errors} error(s))"
        suffix = f", {self.errors} error(s)" if self.errors else ""
        return (
            f"judge: mean={self.mean_score:.0f}/100 "
            f"pass_rate={self.pass_rate:.0%} n={len(self.scored)}{suffix}"
        )
