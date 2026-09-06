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

from .case import (
    ConversationCase,
    ConversationResult,
    TurnCase,
    TurnResult,
)
from .gate import Gate, GateResult, Severity, Threshold, default_gate
from .metrics import (
    Term,
    ToolCall,
    ToolExpectation,
    TurnLatency,
    Utterance,
    check_language,
    confidence_gate,
    term_recall,
    word_error_rate,
)
from .suite import Suite, SuiteReport

__version__ = "0.1.0"

__all__ = [
    "Suite", "SuiteReport",
    "TurnCase", "TurnResult", "ConversationCase", "ConversationResult",
    "Gate", "GateResult", "Threshold", "Severity", "default_gate",
    "Term", "TurnLatency", "Utterance", "ToolCall", "ToolExpectation",
    "word_error_rate", "term_recall", "check_language", "confidence_gate",
]
