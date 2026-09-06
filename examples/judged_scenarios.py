"""Add an LLM judge to scenario evaluation.

Deterministic checks answer "did the agent do the right thing". A judge answers
"was it any good". Run both: the deterministic checks block the deploy, the judge
warns.

    pip install voxeval google-genai
    export GOOGLE_API_KEY=...
    python examples/judged_scenarios.py

Two rules, both learned by getting them wrong against a live agent:

**Never ask the judge something the trace already answers.** A rubric saying the
agent should "call placeOrder exactly once", paired with a transcript that
(correctly) contains no tool lines, made the judge infer from wording that the
tool had not fired — scoring a perfectly correct call 40/100. Tool assertions
belong in `ToolExpectation`.

**Watch for rubric phrases that forbid correct behaviour.** "Quotes prices only
from the menu it was given" sounds precise. It cost an agent 35 points for
correctly adding two menu prices into a total.
"""

from __future__ import annotations

import os

from voxeval import (
    CallTrace,
    Scenario,
    ToolCall,
    ToolExpectation,
    llm_judge,
    run_scenarios,
)


def gemini_judge(model: str = "gemini-3.1-flash-lite"):
    """Wrap any provider in a prompt-in / text-out function.

    `llm_judge` only needs `str -> str`, so swapping providers is this function and
    nothing else — no library change, no adapter class.

    Temperature 0 matters: at 0 the judge reproduces exactly (the same transcript
    scored identically eight times out of eight in testing). Score movement then
    means the *agent* changed, which is the signal you want.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def complete(prompt: str) -> str:
        reply = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=600),
        )
        return reply.text or ""

    return llm_judge(complete)


SCENARIOS = [
    Scenario(
        id="delivery_request",
        caller_turns=["I'd like this delivered to my house, please."],
        description="Caller insists on delivery from a pickup-only restaurant.",
        # Deterministic: an order must NOT be placed as if delivery were arranged.
        expected_tools=ToolExpectation(forbidden=("placeOrder",)),
        # Judged: was the refusal actually good? No mention of tools.
        expected_behavior=(
            "Declines delivery clearly and politely, offers pickup instead, and never "
            "tells the caller that delivery has been arranged."
        ),
    ),
    Scenario(
        id="off_menu_item",
        caller_turns=["Can I get the lobster thermidor?"],
        description="Caller asks for an item that is not on the menu.",
        expected_tools=ToolExpectation(forbidden=("placeOrder",)),
        expected_behavior=(
            "Says plainly that the item is not available and offers what is. Inventing "
            "a price or accepting the order is the worst outcome."
        ),
    ),
]


# Canned replies so the example runs without an agent. Note they are keyed by the
# caller's turn: a stub that ignores `caller_turns` and returns one fixed trace
# will score every scenario against the same transcript and report a confident,
# meaningless 100.
_STUB = {
    "I'd like this delivered to my house, please.": (
        "I'm sorry, we're pickup only — but I can have it ready for you in about "
        "twenty minutes if you'd like to come by."
    ),
    "Can I get the lobster thermidor?": (
        "We don't have lobster thermidor, I'm afraid. Our seafood dish today is the "
        "salt and pepper shrimp — would you like that instead?"
    ),
}


def my_agent(caller_turns) -> CallTrace:
    """Replace with a call into your own agent.

    Returning a `CallTrace` is usually reshaping logs you already have rather than
    instrumenting anything new.
    """
    opening = caller_turns[0]
    return CallTrace(
        transcript=f"Customer: {opening}\nAgent: {_STUB[opening]}",
        tool_calls=[ToolCall(name="checkHours")],
    )


if __name__ == "__main__":
    report = run_scenarios(SCENARIOS, my_agent, judge=gemini_judge())
    print(report)
    print()

    result = report.gate()
    print(result)

    # Only the deterministic checks decide the exit code; `judge_score` and
    # `judge_pass_rate` are WARN thresholds and appear in the output without
    # blocking. Promote them to FAIL once you have measured the end-to-end spread
    # on your own scenarios.
    raise SystemExit(result.exit_code())
