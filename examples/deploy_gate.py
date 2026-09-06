"""A runnable pre-deploy gate.

    python examples/deploy_gate.py          # exits 0
    python examples/deploy_gate.py --broken # exits 1

The --broken run is the point: every failure it reports would pass a
transcript-only evaluation.
"""
import sys

from voxeval import (
    ConversationCase,
    Suite,
    Term,
    ToolCall,
    ToolExpectation,
    TurnCase,
    TurnLatency,
    Utterance,
)

BROKEN = "--broken" in sys.argv

MENU = [
    Term("Kung Pao Chicken", variants=("kung pao", "gong bao", "宫保"),
         collapses=("compound chicken", "compound")),
    Term("Beef with Broccoli", variants=("beef and broccoli",)),
    Term("Spring Roll", variants=("spring rolls",)),
]

suite = Suite("pre-deploy")

suite.add_turn(TurnCase(
    id="order:english",
    reference="I want one kung pao chicken and two spring rolls",
    hypothesis=("I want one compound chicken and two spring rolls" if BROKEN
                else "I want one kung pao chicken and two spring rolls"),
    terms=MENU[:1] + MENU[2:],
    latency=TurnLatency(vad_to_stt_ms=190, stt_to_llm_ms=430, llm_to_tts_ms=95),
    languages=["en", "zh"],
))

suite.add_turn(TurnCase(
    id="order:mandarin",
    reference="我要一份宫保鸡丁",
    hypothesis="એંગ એકડો" if BROKEN else "我要一份宫保鸡丁",
    terms=MENU[:1],
    latency=TurnLatency(vad_to_stt_ms=240, stt_to_llm_ms=510, llm_to_tts_ms=110),
    languages=["en", "zh"],
))

suite.add_turn(TurnCase(
    id="callback-number",
    reference="my number is six one seven five five five zero one four two",
    hypothesis="my number is 617 555 0142",
    # Spoken digits vs numerals: WER is 1.0 and the number is identical.
    score_wer=False,
    latency=TurnLatency(vad_to_stt_ms=200, stt_to_llm_ms=390, llm_to_tts_ms=88),
))

suite.add_conversation(ConversationCase(
    id="call:takeout",
    transcript="Your order is all set — you'll get a confirmation text shortly.",
    tool_calls=([ToolCall("placeOrder", {"items": "..."})] if BROKEN else
                [ToolCall("placeOrder", {"items": "..."}), ToolCall("confirmOrder")]),
    expected_tools=ToolExpectation(
        required=("placeOrder", "confirmOrder"),
        commit_tools=("placeOrder", "confirmOrder"),
    ),
    utterances=[
        Utterance("caller", 0, 2400, "I'd like to place an order"),
        Utterance("agent", 2700, 5200, "Sure, what can I get you?"),
        Utterance("caller", 5000, 7000, "Actually, are you still open?"),   # barge-in
        Utterance("agent", 7300, 9000, "We are, until 9pm."),
    ],
))

report = suite.run()
print(report)
print()
result = report.gate()
print(result)
sys.exit(result.exit_code())
