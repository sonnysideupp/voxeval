"""Adapters: point voxeval at a transcriber or an agent instead of hand-building traces."""
from pathlib import Path

import pytest

from voxeval import (
    AudioCase,
    CallTrace,
    Scenario,
    Term,
    ToolCall,
    ToolExpectation,
    Transcription,
    TurnLatency,
    Utterance,
    assert_gate,
    run_audio,
    run_scenarios,
)


class TestRunAudio:
    def test_bare_function_is_a_valid_transcriber(self):
        # The interface must be satisfiable by a lambda — wrapping a provider
        # should not require a subclass.
        report = run_audio(
            [AudioCase(id="a", audio=Path("fake.wav"), reference="hello there")],
            lambda _: "hello there",
        )
        assert report.metrics["wer"] == 0.0

    def test_transcription_object_carries_latency_through(self):
        report = run_audio(
            [AudioCase(id="a", audio=Path("f.wav"), reference="hi")],
            lambda _: Transcription(
                text="hi",
                confidence=0.94,
                latency=TurnLatency(vad_to_stt_ms=150, stt_to_llm_ms=300, llm_to_tts_ms=70),
            ),
        )
        assert report.metrics["end_to_end_p50_ms"] == 520

    def test_terms_and_collapses_are_scored(self):
        report = run_audio(
            [AudioCase(
                id="order",
                audio=Path("f.wav"),
                terms=[Term("Kung Pao Chicken", variants=("kung pao",),
                            collapses=("compound chicken",))],
            )],
            lambda _: "one compound chicken please",
        )
        assert report.metrics["collapses"] == 1
        assert not report.gate().passed

    def test_transcriber_is_called_once_per_case(self):
        seen: list[Path] = []

        def transcriber(path: Path) -> str:
            seen.append(path)
            return "ok"

        run_audio(
            [AudioCase(id="a", audio=Path("1.wav")), AudioCase(id="b", audio=Path("2.wav"))],
            transcriber,
        )
        assert seen == [Path("1.wav"), Path("2.wav")]


class TestRunScenarios:
    def _agent(self, confirm: bool):
        def agent(caller_turns):
            calls = [ToolCall("placeOrder", {"items": "..."})]
            if confirm:
                calls.append(ToolCall("confirmOrder"))
            return CallTrace(
                transcript="Your order is all set.",
                tool_calls=calls,
                utterances=[Utterance("caller", 0, 1200), Utterance("agent", 1500, 3000)],
                latencies=[TurnLatency(end_to_end_ms=620)],
            )
        return agent

    def test_healthy_agent_passes(self):
        report = run_scenarios(
            [Scenario("order", ["I'd like a pizza", "yes that's right"],
                      ToolExpectation(required=("placeOrder", "confirmOrder")))],
            self._agent(confirm=True),
        )
        assert report.gate().passed

    def test_agent_that_never_commits_fails(self):
        report = run_scenarios(
            [Scenario("order", ["I'd like a pizza"],
                      ToolExpectation(required=("placeOrder", "confirmOrder"),
                                      commit_tools=("confirmOrder",)))],
            self._agent(confirm=False),
        )
        assert not report.gate().passed
        assert report.metrics["claimed_without_commit"] == 1

    def test_caller_turns_reach_the_agent(self):
        received = []

        def agent(caller_turns):
            received.extend(caller_turns)
            return CallTrace(transcript="ok")

        run_scenarios([Scenario("s", ["one", "two"])], agent)
        assert received == ["one", "two"]


class TestAssertGate:
    def test_passes_silently(self):
        assert_gate(run_audio([AudioCase("a", Path("f.wav"), reference="hi")], lambda _: "hi"))

    def test_raises_with_the_full_report(self):
        report = run_audio(
            [AudioCase("a", Path("f.wav"), languages=["en"])],
            lambda _: "એંગ એકડો",
        )
        with pytest.raises(AssertionError) as exc:
            assert_gate(report)
        assert "out_of_set_language_rate" in str(exc.value)


class TestStreaming:
    """A batch transcriber cannot report finalization latency or exercise
    endpointing. These cover the streaming path that can."""

    def _wav(self, tmp_path, ms=1000):
        from voxeval.audio import save_wav, silence
        path = tmp_path / "f.wav"
        save_wav(path, silence(ms, 16000))
        return path

    def test_finalization_latency_reaches_the_report(self, tmp_path):
        from voxeval import Final, StreamResult, run_streaming

        def stt(buffer):
            return StreamResult(
                finals=[Final("one kung pao chicken", at_ms=1240.0, confidence=0.91)],
                audio_ended_at_ms=1000.0,
            )

        report = run_streaming([AudioCase("a", self._wav(tmp_path))], stt)
        assert report.metrics["stt_p95_ms"] == 240.0

    def test_empty_result_reports_no_latency_not_zero(self, tmp_path):
        # A recognizer that transcribed nothing must not look like the fastest one.
        from voxeval import StreamResult, run_streaming

        report = run_streaming([AudioCase("a", self._wav(tmp_path))], lambda b: StreamResult())
        assert report.metrics["stt_p95_ms"] is None

    def test_multiple_finals_are_joined(self, tmp_path):
        from voxeval import Final, StreamResult, run_streaming

        def stt(buffer):
            return StreamResult(
                finals=[Final("one kung pao", 900.0), Final("chicken please", 1300.0)],
                audio_ended_at_ms=1000.0,
            )

        report = run_streaming(
            [AudioCase("a", self._wav(tmp_path),
                       terms=[Term("Kung Pao Chicken", variants=("kung pao",))])],
            stt,
        )
        assert report.metrics["term_recall"] == 1.0

    def test_noise_is_mixed_before_streaming(self, tmp_path):
        import array
        import math

        from voxeval import Final, StreamResult, run_streaming
        from voxeval.audio import AudioBuffer, rms

        seen = {}

        def stt(buffer):
            seen["rms"] = rms(buffer)
            return StreamResult(finals=[Final("x", 10.0)], audio_ended_at_ms=0.0)

        from voxeval.audio import save_wav
        path = tmp_path / "speech.wav"
        n = 16000
        save_wav(path, AudioBuffer(
            array.array("h", (int(6000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(n))),
            16000,
        ))
        noise = AudioBuffer(
            array.array("h", (int(6000 * math.sin(2 * math.pi * 900 * i / 16000)) for i in range(n))),
            16000,
        )
        run_streaming([AudioCase("a", path)], stt, noise=noise, snr_db=0)
        assert seen["rms"] > 0


class TestCallTraceTurns:
    def test_agent_can_return_structured_turns(self):
        from voxeval import CallTrace, Scenario, Term, Turn, run_scenarios

        def agent(caller_turns):
            return CallTrace(turns=[
                Turn(spoken="one kung pao chicken",
                     heard="one compound chicken",
                     agent_said="We don't have that.",
                     terms=[Term("kung pao chicken",
                                 collapses=("compound chicken",))]),
            ])

        report = run_scenarios([Scenario(id="s", caller_turns=["hi"])], agent)
        assert report.metrics["collapses"] == 1.0
        assert report.metrics["wer"] is not None

    def test_flat_transcript_agent_still_supported(self):
        from voxeval import CallTrace, Scenario, run_scenarios

        report = run_scenarios(
            [Scenario(id="s", caller_turns=["hi"])],
            lambda turns: CallTrace(transcript="Customer: hi\nAgent: hello"),
        )
        assert report.conversations[0].ok

    def test_empty_trace_is_rejected(self):
        import pytest

        from voxeval import CallTrace
        with pytest.raises(ValueError, match="turns or a transcript"):
            CallTrace()


class TestCallTraceExplicitEmpty:
    def test_explicit_empty_tool_calls_reaches_the_case(self):
        from voxeval import CallTrace, Scenario, ToolCall, ToolExpectation, Turn, run_scenarios

        report = run_scenarios(
            [Scenario(id="s", caller_turns=["hi"],
                      expected_tools=ToolExpectation(required=("confirmOrder",)))],
            lambda turns: CallTrace(
                turns=[Turn(spoken="do it", tool_calls=[ToolCall("confirmOrder")])],
                tool_calls=[],
            ),
        )
        assert report.metrics["tool_call_failures"] == 1.0

    def test_omitted_tool_calls_still_derive(self):
        from voxeval import CallTrace, Scenario, ToolCall, ToolExpectation, Turn, run_scenarios

        report = run_scenarios(
            [Scenario(id="s", caller_turns=["hi"],
                      expected_tools=ToolExpectation(required=("confirmOrder",)))],
            lambda turns: CallTrace(
                turns=[Turn(spoken="do it", tool_calls=[ToolCall("confirmOrder")])]),
        )
        assert report.metrics["tool_call_failures"] == 0.0


class TestInterimFallback:
    """ElevenLabs Scribe v2 emits no FINAL_TRANSCRIPT at its default config. A
    finals-only harness records a working recognizer as transcribing nothing."""

    def _interim_only(self):
        from voxeval import Final, StreamResult
        return StreamResult(
            interims=[Final(text="I'd like a", at_ms=100.0),
                      Final(text="I'd like a beef noodle soup", at_ms=900.0)],
            audio_ended_at_ms=1000.0,
        )

    def test_text_falls_back_to_last_interim(self):
        assert self._interim_only().text == "I'd like a beef noodle soup"

    def test_fallback_is_flagged(self):
        assert self._interim_only().used_interim_fallback

    def test_latency_does_not_fall_back(self):
        # An interim is a running guess, not an end-of-turn decision. Timing it
        # would report a finalization latency for something that never finalized.
        assert self._interim_only().finalization_latency_ms is None

    def test_confidence_does_not_fall_back(self):
        assert self._interim_only().mean_confidence is None

    def test_finals_win_when_present(self):
        from voxeval import Final, StreamResult
        result = StreamResult(
            finals=[Final(text="the real transcript", at_ms=800.0)],
            interims=[Final(text="the rea", at_ms=200.0)],
            audio_ended_at_ms=700.0)
        assert result.text == "the real transcript"
        assert not result.used_interim_fallback
        assert result.finalization_latency_ms == 100.0

    def test_no_events_at_all(self):
        from voxeval import StreamResult
        empty = StreamResult()
        assert empty.text == ""
        assert not empty.used_interim_fallback

    def test_blank_finals_fall_back_and_are_flagged(self):
        # A whitespace-only final transcribed nothing. `text` and the flag must
        # agree that the interim was used, and nothing finalized to time.
        from voxeval import Final, StreamResult
        result = StreamResult(
            finals=[Final(text="  ", at_ms=1500.0, confidence=0.0)],
            interims=[Final(text="one beef noodle soup", at_ms=900.0)],
            audio_ended_at_ms=1000.0)
        assert result.text == "one beef noodle soup"
        assert result.used_interim_fallback
        assert result.finalization_latency_ms is None
        assert result.mean_confidence is None

    def test_trailing_blank_final_does_not_stretch_latency_or_drag_confidence(self):
        # Recognizers commonly flush an empty final at endpoint. The caller heard
        # the answer at the last final that carried words.
        from voxeval import Final, StreamResult
        result = StreamResult(
            finals=[Final(text="yes", at_ms=1300.0, confidence=0.9),
                    Final(text="", at_ms=2400.0, confidence=0.0)],
            audio_ended_at_ms=1000.0)
        assert result.text == "yes"
        assert not result.used_interim_fallback
        assert result.finalization_latency_ms == 300.0
        assert result.mean_confidence == 0.9
