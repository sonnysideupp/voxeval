"""Score a real streaming recognizer, end to end.

This is the example that produced the padding defaults in `voxeval.audio` and the
rubric warnings in `voxeval.metrics.judge`. It is written against Deepgram Flux
via the LiveKit Agents plugin, but the shape is the same for any streaming STT:
implement one callable that pushes frames and timestamps finals.

    pip install voxeval livekit-agents livekit-plugins-deepgram
    export DEEPGRAM_API_KEY=...
    python examples/live_recognizer.py fixtures/*.wav

What to copy from here:

* **Push at 1x real time.** The `await asyncio.sleep(...)` in the loop is not
  politeness. Dump the file as fast as the socket accepts it and the recognizer
  reports a finalization latency no caller would ever experience.
* **Timestamp finals as they arrive**, and record when the last *real* audio
  frame went out. Finalization latency is the gap between those two, and it is
  the number the caller feels as silence.
* **Let `iter_frames` do the padding.** Its defaults were measured; see its
  docstring for the sweep and why under-padding makes a healthy recognizer look
  like it drops sentences.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from voxeval import (
    AudioBuffer,
    AudioCase,
    Final,
    StreamResult,
    Term,
    run_streaming,
)
from voxeval.audio import iter_frames, realtime_schedule

FRAME_MS = 20
TRAILING_SILENCE_MS = 1500.0


def deepgram_flux(model: str = "flux-general-en", keyterms: list[str] | None = None):
    """A StreamingTranscriber backed by Deepgram Flux."""
    from livekit import rtc
    from livekit.agents.stt import SpeechEventType
    from livekit.agents.utils import http_context
    from livekit.plugins import deepgram

    def transcribe(buffer: AudioBuffer) -> StreamResult:
        async def run() -> StreamResult:
            # The plugin expects the shared session a worker job provides; outside
            # the worker, open one explicitly.
            async with http_context.open():
                return await _stream(buffer)

        return asyncio.run(run())

    async def _stream(buffer: AudioBuffer) -> StreamResult:
        stt = deepgram.STTv2(model=model, keyterm=keyterms or [])
        stream = stt.stream()
        finals: list[Final] = []
        started = time.monotonic()

        def elapsed_ms() -> float:
            return (time.monotonic() - started) * 1000.0

        async def collect() -> None:
            async for event in stream:
                if event.type == SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                    alt = event.alternatives[0]
                    if alt.text.strip():
                        finals.append(Final(
                            text=alt.text,
                            at_ms=elapsed_ms(),
                            confidence=getattr(alt, "confidence", None),
                        ))

        collector = asyncio.create_task(collect())
        samples_per_frame = int(buffer.sample_rate * FRAME_MS / 1000)
        sleep_s = realtime_schedule(FRAME_MS)

        def push(chunk: bytes) -> None:
            stream.push_frame(rtc.AudioFrame(
                data=chunk,
                sample_rate=buffer.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_frame,
            ))

        # Speech (with lead-in), paced at real time.
        for chunk in iter_frames(buffer, FRAME_MS, trailing_silence_ms=0.0):
            push(chunk)
            await asyncio.sleep(sleep_s)

        # Mark end of speech *before* the trailing silence. Latency measured from
        # here is what the caller waits; measuring from end_input() would credit
        # the recognizer for the silence we chose to send.
        audio_ended = elapsed_ms()

        silent_frame = b"\x00" * (samples_per_frame * 2)
        for _ in range(int(TRAILING_SILENCE_MS / FRAME_MS)):
            push(silent_frame)
            await asyncio.sleep(sleep_s)

        stream.end_input()
        try:
            await asyncio.wait_for(collector, timeout=15.0)
        except (TimeoutError, asyncio.TimeoutError):
            collector.cancel()
        finally:
            await stream.aclose()

        return StreamResult(finals=finals, audio_ended_at_ms=audio_ended)

    return transcribe


# Terms are the nouns that carry the transaction. `collapses` are mis-hears you
# have actually observed — detecting one names the failure instead of nudging a
# WER average.
TERMS = [
    Term("kung pao chicken",
         variants=("kung pao", "gong bao", "宫保鸡丁"),
         collapses=("compound chicken", "kong pao")),
    Term("beef with broccoli", variants=("beef and broccoli",)),
]


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 2

    cases = [
        AudioCase(
            id=Path(p).stem,
            audio=Path(p),
            # Drop a sidecar .txt next to each .wav with what was actually said.
            reference=(Path(p).with_suffix(".txt").read_text().strip()
                       if Path(p).with_suffix(".txt").exists() else None),
            terms=TERMS,
            languages=["en"],
        )
        for p in paths
    ]

    report = run_streaming(cases, deepgram_flux())
    print(report)
    print()
    result = report.gate()
    print(result)
    return result.exit_code()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
