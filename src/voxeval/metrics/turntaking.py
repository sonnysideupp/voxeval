"""Turn-taking — did the agent speak at the *right time*?

Two agents can produce identical transcripts and feel completely different on a
call. One waits a beat and answers; the other talks over the caller, or leaves two
seconds of silence so the caller says "hello?" and both start at once.

None of that shows up in transcription accuracy or in end-to-end latency, because
it is about *overlap* and *gaps* between speaker turns rather than about how fast
any single response was.

Three things are worth measuring:

  interruptions   the agent started while the caller was still speaking
  dead air        the caller stopped and nothing happened for too long
  barge-in        the caller interrupted; did the agent actually stop, and how fast

Barge-in recovery is the one users notice most and the one most often broken. An
agent that keeps talking for 800ms after the caller interrupts reads as not
listening, and callers escalate or hang up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


@dataclass(frozen=True)
class Utterance:
    """One speaker's turn on a shared timeline, in milliseconds from call start.

    `speaker` is "caller" or "agent". Timestamps must come from the same clock —
    mixing a client-side and server-side clock produces overlap numbers that are
    pure artefact.
    """

    speaker: str
    start_ms: float
    end_ms: float
    text: str = ""

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


def _overlap_ms(a: Utterance, b: Utterance) -> float:
    return max(0.0, min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms))


@dataclass
class TurnTakingReport:
    interruptions: int = 0
    interruption_ms_total: float = 0.0
    dead_air_events: int = 0
    longest_gap_ms: float = 0.0
    mean_response_gap_ms: float | None = None
    barge_ins: int = 0
    mean_barge_in_recovery_ms: float | None = None
    worst_barge_in_recovery_ms: float | None = None
    unrecovered_barge_ins: int = field(default=0)

    def __str__(self) -> str:
        parts = [
            f"interruptions={self.interruptions}",
            f"dead_air={self.dead_air_events}",
            f"longest_gap={self.longest_gap_ms:.0f}ms",
        ]
        if self.mean_response_gap_ms is not None:
            parts.append(f"mean_gap={self.mean_response_gap_ms:.0f}ms")
        if self.barge_ins:
            recovery = (
                f"{self.mean_barge_in_recovery_ms:.0f}ms"
                if self.mean_barge_in_recovery_ms is not None
                else "n/a"
            )
            parts.append(f"barge_ins={self.barge_ins} mean_recovery={recovery}")
        if self.unrecovered_barge_ins:
            parts.append(f"UNRECOVERED={self.unrecovered_barge_ins}")
        return "  ".join(parts)


def analyze_turns(
    utterances: list[Utterance],
    dead_air_threshold_ms: float = 2000.0,
    interruption_tolerance_ms: float = 100.0,
) -> TurnTakingReport:
    """Measure interruptions, dead air and barge-in recovery over one call.

    `interruption_tolerance_ms` forgives small overlaps. Natural conversation has
    some — humans routinely start ~100ms before the other person finishes, and
    flagging that as a defect would make every healthy call look broken.

    A "barge-in" is the caller starting while the agent is speaking. Recovery is
    how long the agent kept talking afterwards. Anything over roughly 500ms is
    perceived as the agent ignoring the interruption.
    """
    if not utterances:
        return TurnTakingReport()

    ordered = sorted(utterances, key=lambda u: u.start_ms)
    report = TurnTakingReport()
    response_gaps: list[float] = []
    recoveries: list[float] = []

    for i in range(1, len(ordered)):
        prev, curr = ordered[i - 1], ordered[i]
        if prev.speaker == curr.speaker:
            continue

        overlap = _overlap_ms(prev, curr)

        if overlap > interruption_tolerance_ms:
            if curr.speaker == "agent":
                # Agent talked over the caller.
                report.interruptions += 1
                report.interruption_ms_total += overlap
            else:
                # Caller barged in on the agent. Recovery is how much longer the
                # agent kept going after the caller started.
                report.barge_ins += 1
                recovery = max(0.0, prev.end_ms - curr.start_ms)
                recoveries.append(recovery)
                if recovery > 500.0:
                    report.unrecovered_barge_ins += 1
        else:
            gap = curr.start_ms - prev.end_ms
            report.longest_gap_ms = max(report.longest_gap_ms, gap)
            if curr.speaker == "agent":
                # Only caller→agent gaps are response latency. An agent→caller gap
                # is the human thinking, which is not the agent's fault.
                response_gaps.append(gap)
                if gap > dead_air_threshold_ms:
                    report.dead_air_events += 1

    if response_gaps:
        report.mean_response_gap_ms = mean(response_gaps)
    if recoveries:
        report.mean_barge_in_recovery_ms = mean(recoveries)
        report.worst_barge_in_recovery_ms = max(recoveries)
    return report
