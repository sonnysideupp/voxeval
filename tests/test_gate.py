import pytest

from voxeval.gate import Gate, Severity, Threshold, default_gate


class TestThreshold:
    def test_requires_exactly_one_bound(self):
        with pytest.raises(ValueError):
            Threshold("x")
        with pytest.raises(ValueError):
            Threshold("x", max_value=1, min_value=0)

    def test_rejects_negative_margin(self):
        with pytest.raises(ValueError):
            Threshold("x", max_value=1, margin=-0.1)

    def test_margin_widens_in_the_permissive_direction(self):
        # The bug this prevents: `new <= old` with a perfect baseline demands
        # perfection forever, and one token of noise aborts a deploy.
        assert Threshold("wer", max_value=0.10, margin=0.05).evaluate(0.14).passed
        assert not Threshold("wer", max_value=0.10, margin=0.05).evaluate(0.16).passed

    def test_min_value_margin_also_permissive(self):
        assert Threshold("recall", min_value=0.90, margin=0.05).evaluate(0.86).passed
        assert not Threshold("recall", min_value=0.90, margin=0.05).evaluate(0.84).passed

    def test_missing_value_passes_and_is_marked_skipped(self):
        # A pipeline that cannot report a stage must not fail a check about it.
        result = Threshold("stt_p95_ms", max_value=500).evaluate(None)
        assert result.passed and result.skipped

    def test_zero_margin_is_the_default(self):
        assert not Threshold("collapses", max_value=0).evaluate(1).passed
        assert Threshold("collapses", max_value=0).evaluate(0).passed


class TestGate:
    def test_passes_when_all_checks_pass(self):
        gate = Gate().add(Threshold("wer", max_value=0.2)).add(Threshold("recall", min_value=0.9))
        result = gate.check({"wer": 0.1, "recall": 1.0})
        assert result.passed
        assert result.exit_code() == 0

    def test_fails_and_exits_nonzero(self):
        gate = Gate().add(Threshold("wer", max_value=0.2))
        result = gate.check({"wer": 0.9})
        assert not result.passed
        assert result.exit_code() == 1

    def test_warning_does_not_block_deploy(self):
        gate = Gate().add(Threshold("x", max_value=1, severity=Severity.WARN))
        result = gate.check({"x": 99})
        assert result.passed
        assert len(result.warnings) == 1
        assert result.exit_code() == 0

    def test_unmeasured_metrics_do_not_block(self):
        result = default_gate().check({})
        assert result.passed


class TestDefaultGate:
    def test_collapse_fails_with_no_tolerance(self):
        # No acceptable rate of confidently mishearing a product name.
        result = default_gate().check({"collapses": 1})
        assert not result.passed

    def test_out_of_set_language_fails(self):
        result = default_gate().check({"out_of_set_language_rate": 0.01})
        assert not result.passed

    def test_duplicate_commit_fails(self):
        result = default_gate().check({"duplicate_commits": 1})
        assert not result.passed

    def test_false_completion_fails(self):
        result = default_gate().check({"claimed_without_commit": 1})
        assert not result.passed

    def test_healthy_run_passes(self):
        result = default_gate().check({
            "term_recall": 1.0,
            "wer": 0.05,
            "collapses": 0,
            "out_of_set_language_rate": 0.0,
            "end_to_end_p50_ms": 800,
            "end_to_end_p95_ms": 2200,
            "barge_in_recovery_p95_ms": 300,
            "duplicate_commits": 0,
            "claimed_without_commit": 0,
        })
        assert result.passed


class TestStageLatenciesAreGated:
    """Regression: stt/llm/tts p95 were aggregated into metrics but no threshold
    consumed them, so a run could measure a 1.7s STT finalization and print PASS."""

    def test_every_computed_latency_metric_has_a_threshold(self):
        from voxeval import Suite, default_gate
        computed = {
            k for k in Suite().run().metrics
            if k.endswith("_ms") and not k.startswith("barge_in")
        }
        gated = {t.name for t in default_gate()._thresholds}
        assert computed <= gated, f"computed but ungated: {sorted(computed - gated)}"

    def test_slow_stt_finalization_warns(self):
        from voxeval import default_gate
        result = default_gate().check({"stt_p95_ms": 1811.0})
        assert result.passed, "stage latency is advisory, not a deploy blocker"
        assert any(w.threshold.name == "stt_p95_ms" for w in result.warnings)

    def test_healthy_stt_finalization_is_silent(self):
        from voxeval import default_gate
        result = default_gate().check({"stt_p95_ms": 250.0})
        assert not any(w.threshold.name == "stt_p95_ms" for w in result.warnings)
