"""Audio fixtures — load, mix noise, and stream at real time.

Evaluating a voice agent on real audio is not the same as calling
`transcribe(path)`. Two things only exist when you push audio through a live
stream the way a phone call does:

**Finalization latency.** The number the caller feels is *end of speech → final
transcript*, and a batch API cannot report it — you get text with no timeline.
Streaming at 1× real time and timestamping each final is the only way to measure
it.

**Endpointing behaviour.** Turn-taking models decide a turn has ended from
silence. Feed them a file at 4× speed, or with no trailing silence, and they
either never emit a final or emit one at the wrong moment. A recognizer that
looks broken in a batch harness is often fine on a real call, and vice versa.

Everything here is stdlib — `wave`, `array`, `math`. No numpy, so the library
still installs anywhere and its tests still run without a build toolchain.
"""

from __future__ import annotations

import array
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class AudioBuffer:
    """16-bit mono PCM. The format phone audio arrives in, and the one every
    streaming STT accepts."""

    samples: array.array  # signed 16-bit
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.typecode != "h":
            raise ValueError("samples must be a signed 16-bit array ('h')")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    @property
    def duration_ms(self) -> float:
        return 1000.0 * len(self.samples) / self.sample_rate

    def to_bytes(self) -> bytes:
        return self.samples.tobytes()

    def __len__(self) -> int:
        return len(self.samples)


def load_wav(path: Path | str) -> AudioBuffer:
    """Read a mono 16-bit WAV.

    Raises on stereo or non-16-bit input rather than silently converting.
    A downmix that happens by accident changes what you are measuring, and a
    surprising resample is worse than an error — it shifts WER without
    explanation.
    """
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono, got {wf.getnchannels()} channels")
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit, got {wf.getsampwidth() * 8}-bit")
        samples = array.array("h")
        samples.frombytes(wf.readframes(wf.getnframes()))
        return AudioBuffer(samples, wf.getframerate())


def save_wav(path: Path | str, buffer: AudioBuffer) -> None:
    """Write a mono 16-bit WAV. Useful for saving the mixed/denoised audio a
    failing case produced, so you can listen to what the recognizer heard."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(buffer.sample_rate)
        wf.writeframes(buffer.to_bytes())


def rms(buffer: AudioBuffer) -> float:
    if not len(buffer.samples):
        return 0.0
    total = sum(float(s) * float(s) for s in buffer.samples)
    return math.sqrt(total / len(buffer.samples))


def mix_at_snr(speech: AudioBuffer, noise: AudioBuffer, snr_db: float) -> AudioBuffer:
    """Mix noise into speech at a target signal-to-noise ratio.

    Noise is tiled or truncated to the speech length, then scaled so the result
    sits at `snr_db`. The mix is peak-normalized only if it would clip, because
    normalizing unconditionally changes the level of every fixture and makes runs
    incomparable.

    Use this to test the thing that actually breaks calls: a recognizer that is
    perfect in a quiet room and loses the product name over a busy dining room.
    """
    if speech.sample_rate != noise.sample_rate:
        raise ValueError(
            f"sample rate mismatch: speech {speech.sample_rate} vs noise {noise.sample_rate}"
        )
    if not len(speech.samples):
        return speech

    speech_rms, noise_rms = rms(speech), rms(noise)
    if noise_rms == 0:
        return speech

    scale = (speech_rms / (10 ** (snr_db / 20.0))) / noise_rms

    n = len(speech.samples)
    mixed = array.array("h", bytes(2 * n))
    peak = 0
    for i in range(n):
        value = speech.samples[i] + int(noise.samples[i % len(noise.samples)] * scale)
        mixed[i] = max(-32768, min(32767, value))
        peak = max(peak, abs(value))

    if peak > 32767:
        # Only rescale when the sum actually clipped.
        factor = 32767 / peak
        for i in range(n):
            mixed[i] = int(mixed[i] * factor)
    return AudioBuffer(mixed, speech.sample_rate)


def silence(duration_ms: float, sample_rate: int) -> AudioBuffer:
    n = int(sample_rate * duration_ms / 1000.0)
    return AudioBuffer(array.array("h", bytes(2 * n)), sample_rate)


def concat(*buffers: AudioBuffer) -> AudioBuffer:
    if not buffers:
        raise ValueError("concat needs at least one buffer")
    rate = buffers[0].sample_rate
    if any(b.sample_rate != rate for b in buffers):
        raise ValueError("cannot concatenate buffers with different sample rates")
    out = array.array("h")
    for b in buffers:
        out.extend(b.samples)
    return AudioBuffer(out, rate)


def iter_frames(
    buffer: AudioBuffer,
    frame_ms: int = 20,
    trailing_silence_ms: float = 500.0,
) -> Iterator[bytes]:
    """Yield fixed-size PCM frames, padded with trailing silence.

    20ms is the frame size WebRTC and most telephony stacks use, so it is what a
    recognizer expects to receive.

    The trailing silence is not optional padding — it is how a turn-taking model
    learns the turn ended. Without it, endpointing recognizers hold the final
    indefinitely and the case looks like a total transcription failure when the
    audio was fine. Increase it for models that need a longer end-of-turn window.
    """
    frame_samples = int(buffer.sample_rate * frame_ms / 1000)
    if frame_samples <= 0:
        raise ValueError("frame_ms too small for this sample rate")

    padded = concat(buffer, silence(trailing_silence_ms, buffer.sample_rate))
    total = len(padded.samples)
    for start in range(0, total, frame_samples):
        chunk = padded.samples[start:start + frame_samples]
        if len(chunk) < frame_samples:
            chunk = chunk + array.array("h", bytes(2 * (frame_samples - len(chunk))))
        yield chunk.tobytes()


def realtime_schedule(frame_ms: int = 20) -> float:
    """Seconds to sleep between frames to push at 1× real time.

    Pushing faster is a common shortcut and it invalidates every latency number
    in the run: a recognizer that receives ten seconds of audio in two seconds
    reports a finalization delay that no caller would ever experience. Feed it at
    the speed a phone call does.
    """
    return frame_ms / 1000.0
