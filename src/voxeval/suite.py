"""Suite — run cases, aggregate to the metrics a gate consumes.

The aggregation step is where a lot of voice-agent evaluation quietly goes wrong,
so the choices here are explicit:

* **Term recall is averaged per case, not pooled across all terms.** Pooling lets
  one case with thirty terms drown out five cases with two. Per-case averaging
  treats every call as one call, which is how the customer experiences it.

* **Collapses and out-of-set languages are counted, not rated.** A rate invites a
  threshold like "under 5%", and there is no acceptable rate of the agent
  confidently mishearing a product name.

* **Latency is aggregated across every turn in the suite**, not per case then
  averaged, because a percentile of per-case percentiles is not a percentile of
  anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .case import ConversationCase, ConversationResult, TurnCase, TurnResult
from .gate import Gate, GateResult, default_gate
from .metrics.latency import LatencyReport, analyze, percentile


@dataclass
class SuiteReport:
    turns: list[TurnResult] = field(default_factory=list)
    conversations: list[ConversationResult] = field(default_factory=list)
    latency: LatencyReport | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)

    @property
    def failed_turns(self) -> list[TurnResult]:
        return [t for t in self.turns if not t.ok]

    @property
    def failed_conversations(self) -> list[ConversationResult]:
        return [c for c in self.conversations if not c.ok]

    def gate(self, gate: Gate | None = None) -> GateResult:
        return (gate or default_gate()).check(self.metrics)

    def __str__(self) -> str:
        lines: list[str] = []
        if self.turns:
            lines.append(f"Turns ({len(self.turns)}):")
            lines += [f"  {t}" for t in self.turns]
        if self.conversations:
            lines.append(f"\nConversations ({len(self.conversations)}):")
            lines += [f"  {c}" for c in self.conversations]
        if self.latency and self.latency.end_to_end.count:
            lines.append("\nLatency:")
            lines.append(str(self.latency))
        return "\n".join(lines)


class Suite:
    """Collect cases, run them, aggregate.

        suite = Suite("pre-deploy")
        suite.add_turn(TurnCase(id="order", hypothesis=..., terms=[...]))
        report = suite.run()
        result = report.gate()
        sys.exit(result.exit_code())
    """

    def __init__(self, name: str = "voice-agent suite") -> None:
        self.name = name
        self._turns: list[TurnCase] = []
        self._conversations: list[ConversationCase] = []

    def add_turn(self, case: TurnCase) -> "Suite":
        self._turns.append(case)
        return self

    def add_conversation(self, case: ConversationCase) -> "Suite":
        self._conversations.append(case)
        return self

    def run(self) -> SuiteReport:
        turns = [c.evaluate() for c in self._turns]
        conversations = [c.evaluate() for c in self._conversations]

        all_latencies = [c.latency for c in self._turns if c.latency is not None]
        for conv in self._conversations:
            all_latencies.extend(conv.latencies)
        latency = analyze(all_latencies) if all_latencies else None

        return SuiteReport(
            turns=turns,
            conversations=conversations,
            latency=latency,
            metrics=self._aggregate(turns, conversations, latency),
        )

    @staticmethod
    def _aggregate(
        turns: list[TurnResult],
        conversations: list[ConversationResult],
        latency: LatencyReport | None,
    ) -> dict[str, float | None]:
        metrics: dict[str, float | None] = {}

        wers = [t.wer for t in turns if t.wer is not None]
        metrics["wer"] = (sum(wers) / len(wers)) if wers else None

        recalls = [t.terms.recall for t in turns if t.terms is not None]
        metrics["term_recall"] = (sum(recalls) / len(recalls)) if recalls else None

        metrics["collapses"] = float(
            sum(len(t.terms.collapsed) for t in turns if t.terms is not None)
        )

        digits = [t.digits for t in turns if t.digits is not None]
        metrics["digit_accuracy"] = (sum(digits) / len(digits)) if digits else None

        checked = [t for t in turns if t.language is not None]
        metrics["out_of_set_language_rate"] = (
            sum(1 for t in checked if not t.language.ok) / len(checked) if checked else None
        )

        # Every latency key is always present, None when unmeasured. Keys that
        # appear only sometimes make `report.metrics["stt_p95_ms"]` raise for one
        # run and work for the next — the gate tolerates it via .get(), but anyone
        # inspecting the dict directly gets a KeyError that depends on the data.
        metrics["end_to_end_p50_ms"] = latency.end_to_end.p50 if latency else None
        metrics["end_to_end_p95_ms"] = latency.end_to_end.p95 if latency else None
        metrics["stt_p95_ms"] = latency.vad_to_stt.p95 if latency else None
        metrics["llm_ttft_p95_ms"] = latency.stt_to_llm.p95 if latency else None
        metrics["tts_ttfa_p95_ms"] = latency.llm_to_tts.p95 if latency else None

        recoveries = [
            c.turn_taking.worst_barge_in_recovery_ms
            for c in conversations
            if c.turn_taking and c.turn_taking.worst_barge_in_recovery_ms is not None
        ]
        metrics["barge_in_recovery_p95_ms"] = percentile(recoveries, 95) if recoveries else None

        # Missing required tools, forbidden tools, and argument mismatches. These
        # were measured per-conversation but never reached the gate, so a run that
        # never called confirmOrder reported "tool calls incorrect" in the detail
        # and still exited 0.
        metrics["tool_call_failures"] = float(
            sum(1 for c in conversations if c.tools is not None and not c.tools.ok)
        )
        metrics["duplicate_commits"] = float(
            sum(len(c.tools.duplicated) for c in conversations if c.tools is not None)
        )
        metrics["claimed_without_commit"] = float(
            sum(1 for c in conversations if c.false_completion)
        )
        metrics["dead_air_events"] = float(
            sum(c.turn_taking.dead_air_events for c in conversations if c.turn_taking)
        )
        return metrics
