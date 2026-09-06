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
import sys

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

**Voice multi-turn, in one abstraction with text** — a `Turn` carries what the
caller *said*, what the recognizer *heard*, what the agent replied, and what it
did. Voice and text differ in exactly one thing:

```python
Turn(spoken="one kung pao chicken", heard="one compound chicken", ...)   # voice
Turn(spoken="one kung pao chicken", agent_said="Sure.")                  # text
```

Everything downstream — tool traces, latency, judging, turn-taking — is identical.
Transcription metrics switch on precisely when there is a gap to measure, so the
same `ConversationCase` runs for both. **That gap is why this library exists.** A
text-only harness cannot represent it, so it cannot catch a slightly-wrong
transcript making a well-behaved agent do the wrong thing — which is the single
most common way a voice agent fails in production:

```text
mumbled_noisy_environment  3 turns wer=33%  CLAIMED COMPLETION WITHOUT COMMITTING
```

The noisiest call is the one where the agent told the caller their order was
placed and never called the tool. The judge scored that conversation 65/100 — for
an unrelated reason. It never noticed the phantom order, because the transcript
reads like a successful call. Only the tool trace caught it.

**Semantic scoring** — WER compares strings; callers and recognizers disagree
about strings for reasons that don't matter. `"I'd like two coffees at six thirty"`
vs `"I would like 2 coffees at 6:30"` is a *perfect* transcription and a 50% WER.
`semantic_wer` expands contractions, folds spoken numbers to digits and drops
disfluencies first, so the number that survives is one you can gate on. Both are
reported: raw WER says what the recognizer emitted, semantic WER says whether it
mattered. For genuine paraphrase, plug in any embedding model via `Similarity`.

**Conversation quality (LLM judge)** — the questions no assertion can answer: did
it address what the caller asked, invent a price, stay in their language? Opt-in,
provider-agnostic (`llm_judge` takes any `str -> str` function), fails soft, and
defaults to WARN. Errored or abstaining judges are excluded from the mean rather
than counted as zeros — a provider outage shouldn't look like an agent regression.

**Audio fixtures** — WAV and raw-PCM I/O, SNR noise mixing, and 20ms framing for
real-time push, all stdlib (`wave`, `array`, `math`) so the zero-dependency
promise holds.

**Digits** — phone numbers, order numbers, unit numbers. `"five one seven"` and
`"517"` have a WER of 1.0 against each other and are the same number. Scored
separately, and strictly: a phone number that's 90% right is 100% wrong.

**Latency** — decomposed, because the total only tells you a call was slow:

```text
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

## What running it against a live recognizer changed

Every metric here was written against fakes first. Pointing it at a real
streaming recognizer (Deepgram Flux, production config, 1x real-time) changed
three things in the library — which is the point of writing the section down:

**Padding defaults, from a measured sweep.** A healthy 1.9s clip returned an
*empty* transcript. The audio was fine — a batch recognizer transcribed it
perfectly. Sweeping silence padding at both ends (✓ = transcribed, · = empty):

```text
"I'd like to order a beef noodle soup for pickup."   "Yes."   (quiet)
        trail: 500  800 1000 1500          trail: 500  800 1000 1500
lead=   0ms     ·    ·    ·    ✓      lead=   0ms     ✓    ✓    ✓    ✓
lead= 200ms     ·    ·    ✓    ✓      lead= 200ms     ·    ✓    ✓    ✓
lead= 500ms     ✓    ✓    ✓    ✓      lead= 500ms     ·    ·    ·    ✓
```

The two clips want *opposite* lead-ins, and lead-in isn't even monotonic. Under-pad
and you publish "this recognizer silently drops routine sentences" and you're wrong.

The default trailing pad is 2500ms, not 1500ms, for a subtler reason. The quiet
clip's finalization latency was measured at ~1720ms — *longer than 1500ms of
padding*. At 1500ms the final never arrived naturally; the transcript only
appeared because closing the stream flushed it. **If a fixture's latency lands
near your padding, that's the reading to distrust first.**

Running the same sweep against two more recognizers made the rule weaker, and
more honest. **Soniox stt-rt-v4** transcribed everything at every combination.
**ElevenLabs Scribe v2 realtime** was fine on the loud clip but needed
trail ≥ 1500ms on the quiet 0.7s one.

The tempting story after two providers was "turn-taking models need padding,
conventional ones don't." ElevenLabs breaks it — it does no endpointing at all
and is still sensitive on short quiet audio. What the data supports is narrower:
**short, quiet utterances were fragile on two of the three, and the endpointing
one was fragile on both fixtures.** Measure your provider; don't infer from its
category.

ElevenLabs also surfaced a separate trap: **at its default configuration it never
emits `FINAL_TRANSCRIPT`**, only interims. A harness collecting finals alone
records it as transcribing nothing — a working recognizer written up as broken.
`StreamResult` takes interims and falls back to them, flagging it via
`used_interim_fallback`. (With server VAD enabled it does finalize; that mode was
not swept.)

**Latency the transcript can't show.** All seven fixtures transcribed perfectly —
WER 0.00, term recall 1.00. But finalization latency ran a 583ms median with
`"Yes."`, the single most common confirmation in an ordering call, at **1811ms**.
A caller says "yes" and waits nearly two seconds in silence. No text metric sees
this.

**Judges are stable; agents are not.** At temperature 0 the judge scored an
identical transcript 65/100 eight times out of eight, byte-identical reasoning.
Yet one scenario scored 40, 65 and 100 across three end-to-end runs — all of it
agent sampling variance. If your judge scores swing, look at the agent first.

It also caught two rubric bugs worth stealing. Asking the judge whether a tool
fired, while correctly showing it only the spoken transcript, made it infer from
wording and score a correct call 40/100 — tool assertions belong in
`ToolExpectation`. And "quotes prices only from the menu it was given" cost an
agent 35 points for correctly *adding two menu prices into a total*.

See `examples/live_recognizer.py` and `examples/judged_scenarios.py`.

---

## Install

```bash
pip install voxeval
```

**No runtime dependencies.** Every metric is pure Python, so it installs anywhere,
runs in CI without credentials, and the test suite needs no network. Provider SDKs
belong in the harness that produces traces, not in the library that scores them.

---

## Pointing it at your agent

There is no `openai:gpt-4`-style provider string for voice, and no trace format
voice stacks agree on — so `voxeval` asks you for one of two small callables and
does the rest.

**Evaluating the recognizer, streaming** — the one that measures latency. Audio is
pushed at 1× real time with trailing silence, and each final is timestamped:

```python
import time

from voxeval import run_streaming, AudioCase, Term, StreamResult, Final, iter_frames, realtime_schedule

def my_stt(audio):                      # AudioBuffer in, StreamResult out
    stream, finals = deepgram.stream(), []
    t0 = time.perf_counter()
    for frame in iter_frames(audio, frame_ms=20, trailing_silence_ms=500):
        stream.push(frame)
        time.sleep(realtime_schedule(20))      # 1× real time, not as fast as possible
    for f in stream.finals():
        finals.append(Final(f.text, at_ms=(f.received_at - t0) * 1000, confidence=f.confidence))
    return StreamResult(finals=finals, audio_ended_at_ms=audio.duration_ms)

report = run_streaming(cases, my_stt)
report = run_streaming(cases, my_stt, noise=cafe_noise, snr_db=6)   # robustness
```

Two things only exist on this path. **Finalization latency** — end of speech →
final transcript — which a batch API cannot report, because it returns text with
no timeline. And **endpointing**: turn-taking models decide a turn ended from
silence, so pushing at 4× or without trailing silence makes a perfectly good
recognizer emit nothing. Passing `noise` mixes it in at a target SNR first, which
tests the failure that actually loses orders — a recognizer that's flawless in a
quiet room and drops the product name over a busy dining room.

**Evaluating the recognizer, batch** — when you only care about text. A lambda
satisfies it:

```python
import sys
from pathlib import Path

from voxeval import run_audio, AudioCase, Term

cases = [AudioCase(
    id="accented-order",
    audio=Path("fixtures/order.wav"),
    reference="one kung pao chicken",
    terms=[Term("Kung Pao Chicken", variants=("kung pao",), collapses=("compound chicken",))],
    languages=["en", "zh"],
)]

report = run_audio(cases, lambda path: deepgram.transcribe(path).text)
sys.exit(report.gate().exit_code())
```

**Evaluating the conversation** — anything `caller turns → CallTrace`:

```python
from voxeval import run_scenarios, Scenario, CallTrace, ToolCall, ToolExpectation

def my_agent(caller_turns):
    result = replay_conversation(caller_turns)     # your agent, text mode
    return CallTrace(
        transcript=result.transcript,
        tool_calls=[ToolCall(t.name, t.args) for t in result.tools],
        utterances=result.timeline,
        latencies=result.timings,
    )

report = run_scenarios([
    Scenario("takeout", ["I'd like a pizza", "yes, that's right"],
             ToolExpectation(required=("placeOrder", "confirmOrder"),
                             commit_tools=("confirmOrder",))),
], my_agent)
```

Most teams implement `Agent` against a **text-mode replay** rather than real
audio. It runs in CI in seconds with no media stack, and still exercises the
prompt, the tools and the business logic — which is where regressions live.
Evaluate the audio path separately with a `Transcriber`. Splitting the two is
what keeps this runnable on every commit instead of nightly.

### In pytest

```python
from voxeval import assert_gate, run_scenarios

def test_voice_agent_has_not_regressed():
    assert_gate(run_scenarios(SCENARIOS, my_agent))
```

Failures print the whole report, not just the first tripped check.

### Or bring your own traces

The callables are a convenience over the primitive, which is scoring traces you
already have. Build `TurnCase` and `ConversationCase` directly when you're
evaluating recorded production calls rather than running anything.

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

.venv/bin/pytest -q            # 130 tests, no network
.venv/bin/ruff check .         # clean
.venv/bin/mypy src/voxeval     # strict, clean
```

CI runs the tests, ruff and mypy on Python 3.10, 3.11, 3.12 and 3.13. Green on all four.

### Known limitations

Worth stating plainly rather than discovering later:

* **Validated against three recognizers, not many.** Deepgram Flux (which
  produced the padding defaults), Soniox stt-rt-v4 (insensitive to them) and
  ElevenLabs Scribe v2 realtime (sensitive only on short quiet audio, and
  emitting no finals at all). Whisper, Google and Speechmatics have not been
  exercised. Three providers was enough to falsify the clean architectural story
  two suggested, so assume a fourth will surprise you too — measure, don't infer.
* **Semantic normalization is English-only.** The contraction map, disfluency list
  and spoken-number folding are English. `semantic_wer` degrades to ordinary WER
  on other languages rather than failing, but it will not help there. Term recall
  and script detection are language-agnostic and remain the better tools.
* **Morphology is English-only.** `fold_plural` handles regular plurals, `-es`
  after a sibilant, and a table of irregulars ("leaf"/"leaves",
  "knife"/"knives", "child"/"children"). It does not handle other languages, and
  the irregular table is a lookup rather than a full pluralizer — the escape
  hatch for anything it misses is an explicit `Term.variants`. Term matching is
  on token boundaries, so `Term("ham")` will not match "champagne"; an earlier
  substring version did, reporting a term as found when it was never spoken.
* **Number folding is deliberately shallow.** "twenty five" folds to 25; "one
  hundred and twenty three" does not. A wrong fold invents an error in a quantity,
  which is worse than leaving two tokens unmatched.
* **`token_f1` is a floor, not an embedding model.** It cannot see that "can I get"
  and "I'd like to order" mean the same thing. It is genuinely good at the failure
  embeddings are worst at — content that went missing — so it stays useful even
  once you wire up a real `Similarity`.
* **The judge is only as good as the rubric.** Two rubric bugs found in testing
  (see above) each cost a correct agent 35–60 points. Write rubrics as observable
  behaviour, never ask about tool calls, and read the reasoning before believing
  a score.
* **Script detection cannot separate same-alphabet languages.** English from
  Spanish, Hindi from Marathi. It answers "is this plausibly one of my languages",
  not "which language is this".
* **Default thresholds are opinions, not findings.** Calibrate them on your own
  traffic before trusting them.

The test suite is the specification — every metric's edge cases are written down
there, including the ones that motivated it.

## License

MIT
