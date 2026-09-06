"""Add an LLM judge to scenario evaluation.

Deterministic checks answer "did the agent do the right thing". A judge answers
"was it any good". Run both: the deterministic checks block the deploy, the judge
warns.

    pip install voxeval anthropic
    export ANTHROPIC_API_KEY=...
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


def anthropic_judge():
    """Wrap any provider in a prompt-in / text-out function.

    Temperature 0 matters: at 0 the judge reproduces exactly (the same transcript
    scored identically eight times out of eight in testing). Score movement then
    means the *agent* changed, which is the signal you want.
    """
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def complete(prompt: str) -> str:
        reply = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=600,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return reply.content[0].text

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


def my_agent(caller_turns) -> CallTrace:
    """Replace with a call into your own agent.

    Returning a `CallTrace` is usually reshaping logs you already have rather than
    instrumenting anything new.
    """
    return CallTrace(
        transcript=(
            "Customer: I'd like this delivered to my house, please.\n"
            "Agent: I'm sorry, we're pickup only — but I can have it ready for you "
            "in about twenty minutes if you'd like to come by."
        ),
        tool_calls=[ToolCall(name="checkHours")],
    )


if __name__ == "__main__":
    report = run_scenarios(SCENARIOS, my_agent, judge=anthropic_judge())
    print(report)
    print()

    result = report.gate()
    print(result)

    # Only the deterministic checks decide the exit code; `judge_score` and
    # `judge_pass_rate` are WARN thresholds and appear in the output without
    # blocking. Promote them to FAIL once you have measured the end-to-end spread
    # on your own scenarios.
    raise SystemExit(result.exit_code())
