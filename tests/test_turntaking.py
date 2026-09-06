from voxeval.metrics.turntaking import Utterance, analyze_turns


def caller(start, end, text=""):
    return Utterance("caller", start, end, text)


def agent(start, end, text=""):
    return Utterance("agent", start, end, text)


class TestTurnTaking:
    def test_clean_alternation_has_no_problems(self):
        report = analyze_turns([caller(0, 1000), agent(1300, 2500), caller(2800, 3500)])
        assert report.interruptions == 0
        assert report.dead_air_events == 0
        assert report.barge_ins == 0

    def test_small_overlap_is_tolerated(self):
        # Humans routinely start ~100ms early. Flagging that makes healthy calls
        # look broken.
        report = analyze_turns([caller(0, 1000), agent(950, 2000)])
        assert report.interruptions == 0

    def test_agent_talking_over_caller_is_an_interruption(self):
        report = analyze_turns([caller(0, 2000), agent(1000, 3000)])
        assert report.interruptions == 1
        assert report.interruption_ms_total == 1000

    def test_dead_air_detected_only_for_agent_responses(self):
        report = analyze_turns([caller(0, 1000), agent(4000, 5000)])
        assert report.dead_air_events == 1
        assert report.longest_gap_ms == 3000

    def test_caller_thinking_time_is_not_dead_air(self):
        # An agent→caller gap is the human deciding; not the agent's fault.
        report = analyze_turns([agent(0, 1000), caller(9000, 10000)])
        assert report.dead_air_events == 0

    def test_barge_in_recovery_measured(self):
        # Agent speaks 0-5000; caller cuts in at 2000; agent runs on to 5000.
        report = analyze_turns([agent(0, 5000), caller(2000, 3000)])
        assert report.barge_ins == 1
        assert report.worst_barge_in_recovery_ms == 3000
        assert report.unrecovered_barge_ins == 1

    def test_fast_barge_in_recovery_is_not_flagged(self):
        report = analyze_turns([agent(0, 2200), caller(2000, 3000)])
        assert report.barge_ins == 1
        assert report.unrecovered_barge_ins == 0

    def test_empty_input(self):
        assert analyze_turns([]).interruptions == 0

    def test_unsorted_input_is_handled(self):
        report = analyze_turns([agent(1300, 2500), caller(0, 1000)])
        assert report.interruptions == 0


class TestToleratedOverlapIsNotNegativeGap:
    def test_small_overlap_clamps_to_zero(self):
        # A tolerated overlap is not a negative wait. Letting -50ms into the mean
        # reports a faster response time than any call actually had.
        report = analyze_turns([
            Utterance("caller", 0, 1000, "hi"),
            Utterance("agent", 950, 2000, "hello"),
        ], interruption_tolerance_ms=100.0)
        assert report.mean_response_gap_ms is None or report.mean_response_gap_ms >= 0.0
        assert report.longest_gap_ms >= 0.0
