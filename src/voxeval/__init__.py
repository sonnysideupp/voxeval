"""voxeval — evaluate voice agents, not just the text they produce.

Text eval frameworks score what the model said. A voice agent can produce a
perfect sentence and still fail: it misheard the product name, it took four
seconds to answer, it talked over the caller, or it said "you're all set" without
calling the tool that books anything.

This library scores those. It does not run your agent — it scores traces you
already have, so it works against a live recording, a replay harness, or a
synthetic fixture without changing your pipeline.

    from voxeval import Suite, TurnCase, Term

    suite = Suite()
    suite.add_turn(TurnCase(
        id="order",
        reference="I'd like one kung pao chicken",
        hypothesis="I'd like one compound chicken",
        terms=[Term("Kung Pao Chicken", variants=("kung pao",), collapses=("compound",))],
    ))

    report = suite.run()
    sys.exit(report.gate().exit_code())
"""

from .adapters import (
    Agent,
    AudioCase,
    CallTrace,
    Final,
    Scenario,
    StreamingTranscriber,
    StreamResult,
    Transcriber,
    Transcription,
    assert_gate,
    run_audio,
    run_scenarios,
    run_streaming,
)
from .audio import (
    AudioBuffer,
    concat,
    iter_frames,
    load_pcm,
    load_wav,
    mix_at_snr,
    realtime_schedule,
    rms,
    save_wav,
    silence,
)
from .case import (
    ConversationCase,
    ConversationResult,
    TurnCase,
    TurnResult,
)
from .gate import Gate, GateResult, Severity, Threshold, default_gate
from .judges import DEFAULT_TEMPLATE, llm_judge
from .metrics import (
    Judge,
    JudgeReport,
    Similarity,
    Term,
    ToolCall,
    ToolExpectation,
    TurnLatency,
    Utterance,
    Verdict,
    check_language,
    confidence_gate,
    meaning_preserved,
    semantic_normalize,
    semantic_wer,
    term_recall,
    token_f1,
    word_error_rate,
)
from .suite import Suite, SuiteReport

__version__ = "0.1.0"

__all__ = [
    "Suite", "SuiteReport",
    "run_audio", "run_streaming", "run_scenarios", "assert_gate",
    "StreamingTranscriber", "StreamResult", "Final",
    "AudioBuffer", "load_wav", "load_pcm", "save_wav", "mix_at_snr", "iter_frames",
    "silence", "concat", "rms", "realtime_schedule",
    "Transcriber", "Transcription", "Agent", "CallTrace", "AudioCase", "Scenario",
    "TurnCase", "TurnResult", "ConversationCase", "ConversationResult",
    "Gate", "GateResult", "Threshold", "Severity", "default_gate",
    "Term", "TurnLatency", "Utterance", "ToolCall", "ToolExpectation",
    "word_error_rate", "term_recall", "check_language", "confidence_gate",
    "Judge", "JudgeReport", "Verdict", "llm_judge", "DEFAULT_TEMPLATE",
    "Similarity", "semantic_wer", "semantic_normalize", "token_f1", "meaning_preserved",
]
