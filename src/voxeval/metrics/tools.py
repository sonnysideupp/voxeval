"""Tool-call correctness — did the agent *do* the thing, not just say it?

The most expensive voice-agent failures are not bad sentences. They are an agent
that says "your order is all set" without ever calling the tool that places it, or
one that fires the same booking tool twice and charges the customer for two.

A transcript-only evaluation scores both of those as a success, because the words
were perfect. Only the tool trace reveals them.

Three checks, in rough order of how often they catch something real:

  required     the tool that commits the transaction was actually called
  forbidden    a tool that must not fire in this scenario did not
  duplicates   a commit tool fired exactly once, not twice

The duplicate check exists because idempotency bugs in voice agents are
uncommonly easy to introduce — a retry, a re-prompt, or a model that calls the
tool once per item during collection and once more at confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """One observed tool invocation. `arguments` is whatever the model passed,
    already parsed; keep it as the raw dict so argument assertions can inspect
    nested structures without this module knowing your schema."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExpectation:
    """What a scenario expects from the tool trace.

    `commit_tools` are the ones that must fire exactly once when present — the
    place-order, book-appointment, charge-card kind. Listing a tool here is a
    statement that calling it twice costs the customer money.
    """

    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    commit_tools: tuple[str, ...] = ()
    argument_checks: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ToolCallResult:
    ok: bool
    missing: list[str] = field(default_factory=list)
    forbidden_called: list[str] = field(default_factory=list)
    duplicated: list[str] = field(default_factory=list)
    argument_failures: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.ok:
            return "tool calls correct"
        problems = []
        if self.missing:
            problems.append(f"never called {self.missing}")
        if self.forbidden_called:
            problems.append(f"called forbidden {self.forbidden_called}")
        if self.duplicated:
            problems.append(f"DOUBLE-FIRED {self.duplicated}")
        if self.argument_failures:
            problems.append("; ".join(self.argument_failures))
        return " | ".join(problems)


def check_tool_calls(calls: list[ToolCall], expected: ToolExpectation) -> ToolCallResult:
    """Verify a tool trace against a scenario's expectations.

    Argument checks compare only the keys you specify, so a scenario can assert
    that `book_appointment` received `date="Friday"` without having to spell out
    every other field the model may legitimately fill in.
    """
    names = [c.name for c in calls]
    counts: dict[str, int] = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1

    missing = [t for t in expected.required if t not in counts]
    forbidden_called = [t for t in expected.forbidden if t in counts]
    duplicated = [t for t in expected.commit_tools if counts.get(t, 0) > 1]

    argument_failures: list[str] = []
    for tool_name, expected_args in expected.argument_checks.items():
        matching = [c for c in calls if c.name == tool_name]
        if not matching:
            # Absence is reported by the `required` check; don't double-report it.
            continue
        # Pass if ANY call to this tool satisfies the assertion — a model may call
        # a lookup tool several times before the one that matters.
        if not any(
            # `k in` first: .get() returns None for an absent key, so an
            # assertion of {"confirmation_id": None} would pass on a call that
            # never carried the argument at all.
            all(k in call.arguments and call.arguments[k] == v
                for k, v in expected_args.items())
            for call in matching
        ):
            got = [{k: c.arguments.get(k) for k in expected_args} for c in matching]
            argument_failures.append(f"{tool_name} args {expected_args} not satisfied, saw {got}")

    return ToolCallResult(
        ok=not (missing or forbidden_called or duplicated or argument_failures),
        missing=missing,
        forbidden_called=forbidden_called,
        duplicated=duplicated,
        argument_failures=argument_failures,
    )


def claimed_without_committing(
    transcript: str,
    calls: list[ToolCall],
    commit_tools: tuple[str, ...],
    claim_phrases: tuple[str, ...] = (
        "all set", "you're booked", "your order is placed", "confirmation text",
        "i've booked", "i have booked", "order is confirmed",
    ),
) -> bool:
    """True when the agent *said* it completed something but never committed it.

    This is the single highest-value assertion in voice-agent evaluation, and it
    is invisible to every transcript-only method: the words are exactly what a
    successful call sounds like. The caller hangs up believing they have a table,
    an order, or an appointment, and nothing exists on the other end.
    """
    spoken = (transcript or "").lower()
    claimed = any(p in spoken for p in claim_phrases)
    committed = any(c.name in commit_tools for c in calls)
    return claimed and not committed
