# voxeval

**Evaluate voice agents, not just the text they produce.**

A voice agent can produce a flawless transcript and still be broken. It misheard
the product name. It took four seconds to answer. It talked over the caller. It
said *"you're all set"* without ever calling the tool that books anything.

Text evaluation frameworks — LangSmith, Braintrust, Langfuse, DeepEval — score the
model's output. They were not built to catch any of the above, because none of it
is visible in the text.

`voxeval` scores the things that are specific to voice, and turns them into a
pass/fail your deploy script can act on.

```python
from voxeval import Suite, TurnCase, Term

suite = Suite()
suite.add_turn(TurnCase(
    id="order",
    reference="I want one kung pao chicken extra spicy",
    hypothesis="I want one compound chicken extra spicy",   # what the recognizer heard
    terms=[Term("Kung Pao Chicken",
                variants=("kung pao", "gong bao", "宫保"),
                collapses=("compound chicken",))],
))

report = suite.run()
print(report.metrics["wer"])        # 0.125  — looks fine
print(report.metrics["collapses"])  # 1.0    — the item silently left the order
sys.exit(report.gate().exit_code()) # 1      — blocked
```

That example is the whole thesis. **WER said the call was fine. The customer got
the wrong food.**

---

## Why this exists

This started as the pre-deploy gate for a production restaurant voice agent — real
callers, real orders, real money. Every metric here exists because something broke
in production and a text-based eval had scored it as a pass.

| what happened | why text eval missed it |
|---|---|
| "Kung Pao Chicken" → "compound chicken"; the item silently left the order | WER was low; most words were right |
| A noise burst transcribed as fluent Chinese on an English-only line, and the agent replied to it | No reference to score against — the sentence was well-formed |
| An English-only agent returned Gujarati script for "egg roll", five times in one call | Same: confident output, wrong language, no WER signal |
| The agent said "your order is all set" but never called `confirmOrder` | The transcript is *exactly* what success sounds like |
| A commit tool fired twice on one order | Both calls succeeded; nothing in the text is wrong |
| A caller's name at 0.61 confidence was dropped by an over-tight gate, so the agent kept re-asking | Not a text problem at all |

---

## What it measures

**Transcription** — WER, plus **term recall**: did the words that carry the
transaction survive? Dish names, street names, drug names, part numbers. Declare
accepted `variants` (orthography and script differences aren't errors) and known
`collapses` (specific mis-hears worth failing on by name).

**Digits** — phone numbers, order numbers, unit numbers. `"five one seven"` and
`"517"` have a WER of 1.0 against each other and are the same number. Scored
separately, and strictly: a phone number that's 90% right is 100% wrong.

**Latency** — decomposed, because the total only tells you a call was slow:

```
end of speech  →  transcript     STT finalization
transcript     →  first token    LLM TTFT
first token    →  first audio    TTS TTFA
end of speech  →  first audio    what the caller actually feels
```

Percentiles, not averages — averages hide the tail the caller remembers as dead
air. The report names the bottleneck stage by p95.

**Language enforcement** — script-based detection of transcripts in languages your
agent doesn't support. Cheap, deterministic, and it catches the failure mode where
the recognizer invents a fluent sentence nobody said.

**Turn-taking** — interruptions, dead air, and **barge-in recovery**: the caller
cut in, did the agent actually stop, and how fast? Anything over ~500ms reads as
not listening.

**Tool calls** — required, forbidden, argument assertions, and duplicate-commit
detection. Plus `claimed_without_committing()`, the highest-value assertion in the
library: the agent *said* it booked something and never did.

---

## Install

```bash
pip install voxeval
```

**No runtime dependencies.** Every metric is pure Python, so it installs anywhere,
runs in CI without credentials, and the test suite needs no network. Provider SDKs
belong in the harness that produces traces, not in the library that scores them.

---

## Using it as a deploy gate

`voxeval` doesn't run your agent. It scores traces you already have — which means
it works against a live call recording, a replay harness, or a synthetic fixture
without changing your pipeline.

```python
import sys
from voxeval import Suite, TurnCase, ConversationCase, Term, ToolCall, ToolExpectation, TurnLatency

suite = Suite("pre-deploy")

for call in load_recent_calls():                     # your traces, your source
    suite.add_turn(TurnCase(
        id=call.id,
        hypothesis=call.transcript,
        terms=[Term(item) for item in catalog_terms()],
        latency=TurnLatency(
            vad_to_stt_ms=call.stt_ms,
            stt_to_llm_ms=call.ttft_ms,
            llm_to_tts_ms=call.ttfa_ms,
        ),
        languages=["en", "es"],
    ))
    suite.add_conversation(ConversationCase(
        id=call.id,
        transcript=call.transcript,
        tool_calls=[ToolCall(t.name, t.args) for t in call.tools],
        expected_tools=ToolExpectation(commit_tools=("confirmOrder",)),
    ))

report = suite.run()
print(report)
print(result := report.gate())
sys.exit(result.exit_code())          # 0 ships, 1 blocks
```

### Thresholds have margins, on purpose

```python
from voxeval import Gate, Threshold, Severity

gate = Gate("my-gate")
gate.add(Threshold("term_recall", min_value=0.90, margin=0.05))
gate.add(Threshold("collapses",   max_value=0))                     # no margin
gate.add(Threshold("barge_in_recovery_p95_ms", max_value=500,
                   severity=Severity.WARN))                          # observe first
```

A check written as `new <= old` looks rigorous and is unusable. When the baseline
is already perfect it demands perfection forever, and a one-token difference —
well inside the measurement's own noise — aborts a deploy. **A gate that fails one
run in three gets ignored, and an ignored gate protects nothing.**

So metrics carry margins, and defects don't. A commit tool firing twice or an
agent claiming "all set" without committing aren't noisy measurements — they're
booleans, and they fail at one occurrence.

`Severity.WARN` records a problem without blocking. Use it while calibrating a new
check against real traffic, then promote it to `FAIL` once you know its
false-positive rate. Shipping a check straight to `FAIL` is how gates get disabled.

### Missing metrics pass

A speech-to-speech model has no discrete transcript, so it can't report STT
latency. That should degrade the report, not fail a check it cannot answer.
Absence of evidence isn't a regression.

---

## How this relates to the existing work

Voice agent evaluation has good academic benchmarks — **VoiceBench**,
**EVA-Bench**, **VoiceAgentBench**, **AU-Harness** — and good commercial
observability products.

They solve a different problem. Benchmarks compare *models* on fixed datasets.
Observability platforms give you dashboards over production traffic.

`voxeval` is neither. It's a **library you run in CI against your own agent**, on
your own domain terms, that exits non-zero when a regression would reach a
customer. The closest analogy is `pytest` rather than a leaderboard.

---

## Development

```bash
git clone https://github.com/<you>/voxeval && cd voxeval
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q          # 97 tests, no network
```

The test suite is the specification — every metric's edge cases are written down
there, including the ones that motivated it.

## License

MIT
