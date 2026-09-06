"""Latency metrics — did the agent answer *fast enough*?

A voice agent's latency budget is not one number. The caller experiences the gap
between finishing their sentence and hearing the first syllable back, and that gap
is the sum of four stages that fail for different reasons and are fixed by
different people:

    end of speech  →  transcript      (STT finalization)
    transcript     →  first token     (LLM TTFT)
    first token    →  first audio     (TTS TTFA)
    end of speech  →  first audio     (what the caller actually feels)

Reporting only the total tells you a call was slow. Reporting the decomposition
tells you the LLM was slow, which is the only version you can act on.

Percentiles, not averages. Averages hide the calls that lose customers: published
figures put median voice-agent latency around 1.4-1.7s while roughly 10% of calls
exceed 3-5s, and it is that tail the caller remembers as dead air.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median


@dataclass(frozen=True)
class TurnLatency:
    """Per-turn stage timings in milliseconds. Any stage may be None when a
    pipeline does not expose it — a speech-to-speech model has no discrete
    transcript, and that should degrade the report rather than crash it."""

    vad_to_stt_ms: float | None = None
    stt_to_llm_ms: float | None = None
    llm_to_tts_ms: float | None = None
    end_to_end_ms: float | None = None

    def resolved_end_to_end(self) -> float | None:
        """Measured end-to-end when available, else the sum of known stages.

        Prefers the measured value because the stages do not necessarily sum to it:
        pipelines overlap work, and a sum would quietly over-report on any
        implementation that streams TTS while the LLM is still generating.
        """
        if self.end_to_end_ms is not None:
            return self.end_to_end_ms
        stages = [self.vad_to_stt_ms, self.stt_to_llm_ms, self.llm_to_tts_ms]
        known = [s for s in stages if s is not None]
        return sum(known) if len(known) == len(stages) else None


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile. `p` in [0, 100]. None for an empty input.

    Nearest-rank rather than interpolated: with the handful of turns a test suite
    produces, an interpolated p95 invents a number that no call actually
    experienced. Every value this returns was really observed.
    """
    if not values:
        return None
    if not 0 <= p <= 100:
        raise ValueError(f"percentile must be in [0, 100], got {p}")
    ordered = sorted(values)
    if p == 0:
        return ordered[0]
    # ceil(n * p / 100) on the float. Truncating p with int() first collapsed
    # p99.9 to p99 — on 100 samples that returns the 99th value instead of the
    # 100th, quietly hiding the exact tail a p99.9 was asked for.
    rank = max(1, min(len(ordered), math.ceil(len(ordered) * p / 100)))
    return ordered[rank - 1]


@dataclass
class LatencyStats:
    count: int
    p50: float | None
    p95: float | None
    p99: float | None
    mean: float | None
    max: float | None

    def __str__(self) -> str:
        if not self.count:
            return "no samples"
        return (
            f"n={self.count} p50={self.p50:.0f}ms p95={self.p95:.0f}ms "
            f"p99={self.p99:.0f}ms max={self.max:.0f}ms"
        )


def summarize(values: list[float]) -> LatencyStats:
    clean = [v for v in values if v is not None]
    if not clean:
        return LatencyStats(0, None, None, None, None, None)
    return LatencyStats(
        count=len(clean),
        p50=median(clean),
        p95=percentile(clean, 95),
        p99=percentile(clean, 99),
        mean=sum(clean) / len(clean),
        max=max(clean),
    )


@dataclass
class LatencyReport:
    """Per-stage summaries across a set of turns."""

    vad_to_stt: LatencyStats
    stt_to_llm: LatencyStats
    llm_to_tts: LatencyStats
    end_to_end: LatencyStats
    slowest_stage_p95: str | None = field(default=None)

    def __str__(self) -> str:
        lines = [
            f"  vad→stt      {self.vad_to_stt}",
            f"  stt→llm      {self.stt_to_llm}",
            f"  llm→tts      {self.llm_to_tts}",
            f"  end-to-end   {self.end_to_end}",
        ]
        if self.slowest_stage_p95:
            lines.append(f"  bottleneck   {self.slowest_stage_p95} (highest p95)")
        return "\n".join(lines)


def analyze(turns: list[TurnLatency]) -> LatencyReport:
    """Summarize turns and name the bottleneck stage.

    The bottleneck is the stage with the highest p95, not the highest mean — you
    optimize for the calls that felt broken, not the ones that felt fine.
    """
    stages = {
        "vad→stt": [t.vad_to_stt_ms for t in turns if t.vad_to_stt_ms is not None],
        "stt→llm": [t.stt_to_llm_ms for t in turns if t.stt_to_llm_ms is not None],
        "llm→tts": [t.llm_to_tts_ms for t in turns if t.llm_to_tts_ms is not None],
    }
    summaries = {name: summarize(vals) for name, vals in stages.items()}

    scored = [(name, s.p95) for name, s in summaries.items() if s.p95 is not None]
    bottleneck = max(scored, key=lambda kv: kv[1])[0] if scored else None

    e2e = [v for v in (t.resolved_end_to_end() for t in turns) if v is not None]
    return LatencyReport(
        vad_to_stt=summaries["vad→stt"],
        stt_to_llm=summaries["stt→llm"],
        llm_to_tts=summaries["llm→tts"],
        end_to_end=summarize(e2e),
        slowest_stage_p95=bottleneck,
    )
