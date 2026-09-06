import pytest

from voxeval.metrics.latency import TurnLatency, analyze, percentile, summarize


class TestPercentile:
    def test_returns_an_observed_value(self):
        # Nearest-rank, not interpolated: every reported number really happened.
        assert percentile([10, 20, 30, 40], 50) in {10, 20, 30, 40}

    def test_p100_is_max_and_p0_is_min(self):
        assert percentile([5, 1, 9], 100) == 9
        assert percentile([5, 1, 9], 0) == 1

    def test_empty_is_none(self):
        assert percentile([], 95) is None

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            percentile([1, 2], 101)


class TestSummarize:
    def test_empty_has_zero_count(self):
        stats = summarize([])
        assert stats.count == 0 and stats.p50 is None

    def test_ignores_none_values(self):
        stats = summarize([100, None, 200])
        assert stats.count == 2

    def test_computes_max_and_mean(self):
        stats = summarize([100, 200, 300])
        assert stats.max == 300
        assert stats.mean == 200


class TestAnalyze:
    def test_prefers_measured_end_to_end_over_sum(self):
        # Stages overlap in a streaming pipeline; summing over-reports.
        turn = TurnLatency(vad_to_stt_ms=200, stt_to_llm_ms=300, llm_to_tts_ms=100, end_to_end_ms=450)
        assert turn.resolved_end_to_end() == 450

    def test_falls_back_to_sum_when_all_stages_known(self):
        turn = TurnLatency(vad_to_stt_ms=200, stt_to_llm_ms=300, llm_to_tts_ms=100)
        assert turn.resolved_end_to_end() == 600

    def test_returns_none_when_stages_incomplete(self):
        # A partial sum would silently understate; better to report nothing.
        assert TurnLatency(vad_to_stt_ms=200).resolved_end_to_end() is None

    def test_identifies_bottleneck_by_p95(self):
        turns = [
            TurnLatency(vad_to_stt_ms=100, stt_to_llm_ms=900, llm_to_tts_ms=80),
            TurnLatency(vad_to_stt_ms=110, stt_to_llm_ms=950, llm_to_tts_ms=90),
        ]
        assert analyze(turns).slowest_stage_p95 == "stt→llm"

    def test_survives_pipeline_with_no_transcript_stage(self):
        # Speech-to-speech models expose no STT stage. Degrade, don't crash.
        report = analyze([TurnLatency(end_to_end_ms=400), TurnLatency(end_to_end_ms=500)])
        assert report.end_to_end.count == 2
        assert report.vad_to_stt.count == 0


class TestFractionalPercentiles:
    def test_p99_9_is_not_truncated_to_p99(self):
        # int(p) collapsed 99.9 to 99: on 100 samples that returns the 99th
        # value, hiding the exact tail the caller asked about.
        values = [float(i) for i in range(1, 101)]
        assert percentile(values, 99.9) == 100.0
        assert percentile(values, 99) == 99.0

    def test_fractional_below_the_first_rank(self):
        assert percentile([float(i) for i in range(1, 101)], 0.5) == 1.0

    def test_integer_percentiles_are_unchanged(self):
        values = [float(i) for i in range(1, 101)]
        assert percentile(values, 50) == 50.0
        assert percentile(values, 95) == 95.0
