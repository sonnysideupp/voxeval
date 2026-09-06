"""Audio fixtures: WAV I/O, SNR mixing, real-time framing."""
import array
import math

import pytest

from voxeval.audio import (
    AudioBuffer,
    concat,
    iter_frames,
    load_wav,
    mix_at_snr,
    realtime_schedule,
    rms,
    save_wav,
    silence,
)


def tone(freq=440.0, ms=1000.0, rate=16000, amp=8000) -> AudioBuffer:
    n = int(rate * ms / 1000)
    return AudioBuffer(
        array.array("h", (int(amp * math.sin(2 * math.pi * freq * i / rate)) for i in range(n))),
        rate,
    )


class TestAudioBuffer:
    def test_rejects_wrong_sample_width(self):
        with pytest.raises(ValueError):
            AudioBuffer(array.array("i", [1, 2, 3]), 16000)

    def test_rejects_bad_sample_rate(self):
        with pytest.raises(ValueError):
            AudioBuffer(array.array("h", [1]), 0)

    def test_duration(self):
        assert silence(500, 16000).duration_ms == pytest.approx(500)


class TestWavRoundTrip:
    def test_save_then_load_preserves_audio(self, tmp_path):
        original = tone(ms=100)
        path = tmp_path / "t.wav"
        save_wav(path, original)
        loaded = load_wav(path)
        assert loaded.sample_rate == original.sample_rate
        assert list(loaded.samples) == list(original.samples)

    def test_rejects_stereo(self, tmp_path):
        # Silently downmixing changes what you're measuring.
        import wave
        path = tmp_path / "stereo.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 400)
        with pytest.raises(ValueError, match="mono"):
            load_wav(path)


class TestMixAtSnr:
    def test_lower_snr_means_more_noise(self):
        speech, noise = tone(440, 500), tone(1000, 500)
        quiet = mix_at_snr(speech, noise, snr_db=20)
        loud = mix_at_snr(speech, noise, snr_db=0)
        # At 0dB the noise matches the speech level, so the mix deviates further
        # from the original than at 20dB.
        dev_quiet = sum(abs(quiet.samples[i] - speech.samples[i]) for i in range(len(speech)))
        dev_loud = sum(abs(loud.samples[i] - speech.samples[i]) for i in range(len(speech)))
        assert dev_loud > dev_quiet

    def test_achieves_requested_snr(self):
        speech, noise = tone(440, 500), tone(1000, 500)
        scale = (rms(speech) / (10 ** (6 / 20.0))) / rms(noise)
        scaled = AudioBuffer(array.array("h", (int(s * scale) for s in noise.samples)), noise.sample_rate)
        measured = 20 * math.log10(rms(speech) / rms(scaled))
        assert measured == pytest.approx(6.0, abs=0.5)

    def test_rejects_sample_rate_mismatch(self):
        with pytest.raises(ValueError, match="sample rate"):
            mix_at_snr(tone(rate=16000, ms=100), tone(rate=8000, ms=100), 6)

    def test_silent_noise_returns_speech_unchanged(self):
        speech = tone(ms=100)
        mixed = mix_at_snr(speech, silence(100, 16000), 6)
        assert list(mixed.samples) == list(speech.samples)

    def test_shorter_noise_is_tiled(self):
        mixed = mix_at_snr(tone(ms=500), tone(1000, ms=100), 6)
        assert len(mixed) == len(tone(ms=500))

    def test_output_never_clips(self):
        loud = AudioBuffer(array.array("h", [32000] * 1000), 16000)
        mixed = mix_at_snr(loud, tone(1000, ms=63), 0)
        assert all(-32768 <= s <= 32767 for s in mixed.samples)


class TestFraming:
    def test_frames_are_fixed_size(self):
        frames = list(iter_frames(tone(ms=100), frame_ms=20, trailing_silence_ms=0, leading_silence_ms=0))
        assert all(len(f) == 2 * 320 for f in frames)  # 20ms @ 16k, 16-bit

    def test_trailing_silence_is_appended(self):
        without = list(iter_frames(tone(ms=100), 20, trailing_silence_ms=0, leading_silence_ms=0))
        with_pad = list(iter_frames(tone(ms=100), 20, trailing_silence_ms=500, leading_silence_ms=0))
        # Endpointing models need this to decide the turn ended; without it they
        # hold the final and the case looks like a transcription failure.
        assert len(with_pad) == len(without) + 25

    def test_final_partial_frame_is_padded(self):
        frames = list(iter_frames(tone(ms=25), frame_ms=20, trailing_silence_ms=0, leading_silence_ms=0))
        assert len(frames) == 2
        assert len(frames[-1]) == 2 * 320

    def test_realtime_schedule_matches_frame_size(self):
        assert realtime_schedule(20) == 0.020


class TestConcat:
    def test_joins_buffers(self):
        joined = concat(silence(100, 16000), silence(150, 16000))
        assert joined.duration_ms == pytest.approx(250)

    def test_rejects_rate_mismatch(self):
        with pytest.raises(ValueError):
            concat(silence(10, 16000), silence(10, 8000))


class TestLoadPcm:
    """Headerless PCM — what telephony actually carries."""

    def test_roundtrip(self, tmp_path):
        import array

        from voxeval.audio import AudioBuffer, load_pcm
        src = AudioBuffer(array.array("h", [0, 1000, -1000, 32767, -32768]), 16000)
        raw = tmp_path / "a.pcm"
        raw.write_bytes(src.to_bytes())
        loaded = load_pcm(raw, 16000)
        assert list(loaded.samples) == list(src.samples)
        assert loaded.sample_rate == 16000

    def test_rejects_odd_byte_count(self, tmp_path):
        from voxeval.audio import load_pcm
        raw = tmp_path / "bad.pcm"
        raw.write_bytes(b"\x00\x01\x02")
        with pytest.raises(ValueError, match="16-bit"):
            load_pcm(raw, 16000)

    def test_sample_rate_is_required_not_guessed(self, tmp_path):
        # Reading 8kHz call audio as 16kHz transcribes as gibberish and looks
        # like a broken recognizer, so the rate must come from the caller.
        import inspect

        from voxeval.audio import load_pcm
        assert "sample_rate" in inspect.signature(load_pcm).parameters
        assert inspect.signature(load_pcm).parameters["sample_rate"].default is inspect.Parameter.empty


class TestLeadingSilence:
    """Regression: Deepgram Flux returned an empty transcript for a fixture whose
    speech began on the first frame, and transcribed the identical audio
    perfectly with 200ms of lead-in. A real call always carries room tone before
    the caller speaks, so the harness must too."""

    def test_lead_in_is_on_by_default(self):
        from voxeval.audio import iter_frames
        frames = list(iter_frames(tone(ms=100), 20, trailing_silence_ms=0))
        assert len(frames) == (500 + 100) // 20

    def test_default_pads_sit_inside_the_measured_safe_region(self):
        # The sweep in iter_frames' docstring puts lead=200/trail=500 in the
        # FAILING region for Deepgram Flux. Defaults must not sit there, and must
        # not sit on the boundary either — it moves with model and phrase.
        import inspect

        from voxeval.audio import iter_frames
        params = inspect.signature(iter_frames).parameters
        assert params["leading_silence_ms"].default >= 500
        # 1500ms is where both measured fixtures pass at every lead-in tested.
        # Below it the two disagree and there is no setting that satisfies both.
        assert params["trailing_silence_ms"].default >= 1500

    def test_lead_in_frames_are_silent(self):
        import array

        from voxeval.audio import iter_frames
        frames = list(iter_frames(tone(ms=100), 20, trailing_silence_ms=0,
                                  leading_silence_ms=100))
        for frame in frames[:5]:
            assert set(array.array("h", frame)) == {0}

    def test_lead_in_can_be_disabled_for_edge_testing(self):
        from voxeval.audio import iter_frames
        frames = list(iter_frames(tone(ms=100), 20, trailing_silence_ms=0,
                                  leading_silence_ms=0))
        assert len(frames) == 5

    def test_audio_is_preserved_between_the_pads(self):
        import array

        from voxeval.audio import iter_frames
        src = tone(ms=100)
        joined = array.array("h")
        for f in iter_frames(src, 20, trailing_silence_ms=40, leading_silence_ms=60):
            joined.extend(array.array("h", f))
        lead = 60 * 16   # samples at 16kHz
        assert list(joined[lead:lead + len(src.samples)]) == list(src.samples)
