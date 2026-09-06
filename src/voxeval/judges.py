"""A ready-made LLM judge, wrapped around whatever completion function you have.

The valuable part of an LLM judge is not the API call — it is the rubric prompt
and the parsing, both of which are easy to get subtly wrong in ways that show up
as score variance months later. So this module ships those, and takes the model
call as a plain `str -> str` callable:

    from anthropic import Anthropic
    client = Anthropic()

    def complete(prompt: str) -> str:
        reply = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return reply.content[0].text

    judge = llm_judge(complete)

No provider SDK is imported here, so the library stays dependency-free and works
against any model, including a local one.

Three prompt decisions, each of which measurably reduces variance:

* **Score last.** The model writes its observations before committing to a
  number, so the number is a conclusion rather than a first impression it then
  rationalizes.
* **Anchored bands.** "90-100 means X" beats "score out of 100", which drifts
  between model versions and gives you a benchmark that silently rebases.
* **Explicit abstention.** A judge with no way to say "the transcript does not
  show this" will invent a judgement, and an invented judgement is
  indistinguishable from a real one in the aggregate.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from .metrics.judge import Judge, Verdict

DEFAULT_TEMPLATE = """You are evaluating a single phone call handled by an automated voice agent.

# What the caller was trying to do
{scenario}

# What a correct agent does in this situation
{expected}

# Transcript
{transcript}

# Instructions
Judge ONLY what the transcript shows. Do not assume the agent called a tool,
looked something up, or knew a fact unless the transcript demonstrates it.

Ignore transcription artifacts: missing punctuation, lowercase text, run-on
words and minor misspellings come from speech recognition, not from the agent,
and are scored separately elsewhere.

Weigh these, in order:
1. Did the agent accomplish what the caller asked, or correctly decline it?
2. Did it state anything not supported by the context it was given? Inventing a
   price, an address, or an availability is the most serious failure here.
3. Did it confirm the details that matter before acting?
4. Was it concise? A phone caller is waiting in silence while the agent talks.

Scoring bands:
- 90-100: did the right thing; nothing invented; a caller would be satisfied.
- 70-89:  right outcome, with a rough edge — clumsy phrasing, a missed
          confirmation, more words than needed.
- 40-69:  partially handled, or handled correctly but the caller would be
          confused or would have to call back.
- 0-39:   wrong outcome, invented information, or the caller's actual request
          was never addressed.

If the transcript is empty or too short to judge, use "abstain": true.

Reply with ONLY a JSON object:
{{"observations": "<2-3 sentences of what you actually see>",
  "score": <integer 0-100>,
  "abstain": false}}"""


def llm_judge(
    complete: Callable[[str], str],
    template: str = DEFAULT_TEMPLATE,
    pass_score: float = 70.0,
) -> Judge:
    """Build a `Judge` from any prompt-in / text-out function.

    A note on pinning sampling: `temperature=0` is the usual way to make a judge
    reproducible, and it works — but some newer models no longer accept the
    parameter at all (claude-sonnet-5 returns "`temperature` is deprecated for
    this model" and fails the request). If your `complete` passes temperature
    unconditionally, that surfaces here as an errored verdict on *every* case:
    quiet, total loss of the metric rather than a crash. Catch it, retry without,
    and re-measure the score spread — on a model whose sampling you cannot pin,
    reproducibility is no longer free.

    Abstentions come back as errored verdicts, which the suite excludes from the
    mean instead of averaging in as a zero. That distinction matters: a judge
    that could not evaluate three cases is a coverage gap, while three genuine
    zeros is an outage-grade agent regression, and collapsing them into one
    number hides whichever one is real.
    """

    def judge(scenario: str, expected: str, transcript: str) -> Verdict:
        if not (transcript or "").strip():
            return Verdict(score=0.0, error="empty transcript", passed=True)

        raw = complete(template.format(
            scenario=scenario or "(not specified)",
            expected=expected,
            transcript=transcript,
        ))
        payload = _extract_json(raw)
        if payload is None:
            # Treated as unavailable rather than as a zero — an unparseable reply
            # is a fault in the judge, and scoring the agent for it would make a
            # model upgrade look like a regression.
            return Verdict(score=0.0, error=f"unparseable judge reply: {raw[:120]!r}", passed=True)

        abstain = payload.get("abstain", False)
        if not isinstance(abstain, bool):
            # Raw truthiness reads the strings "false" and "no" as abstentions,
            # and abstentions are dropped from the mean — so a model answering in
            # the wrong type silently removes cases from the score.
            return Verdict(score=0.0, passed=True,
                           error=f"non-boolean abstain: {abstain!r}")
        if abstain:
            return Verdict(score=0.0, error="judge abstained", passed=True)

        try:
            score = float(payload["score"])
        except (KeyError, TypeError, ValueError):
            return Verdict(score=0.0, error=f"no usable score in {payload!r}", passed=True)

        score = max(0.0, min(100.0, score))
        return Verdict(
            score=score,
            reasoning=str(payload.get("observations", "")),
            passed=score >= pass_score,
        )

    return judge


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model reply.

    Models wrap JSON in prose or fences however they feel that day, and a judge
    that raises on a stray ```json fence turns a formatting quirk into a failed
    eval run. Try the whole string, then the first braced span.
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    for candidate in (text, _first_braced_span(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _first_braced_span(text: str) -> str | None:
    """The first balanced {...} span, so prose around the JSON is tolerated."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None
