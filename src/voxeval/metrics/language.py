"""Language enforcement — did the recognizer stay inside the languages you support?

A multilingual voice agent is configured for a set of languages. On short, noisy,
or clipped phone audio, recognizers do not simply return a poor transcript in the
right language — they return a confident transcript in a language that was never
spoken and is not in the agent's set.

That failure mode is invisible to word error rate, because there is no sensible
reference to score against. It is also worse than a bad transcript: the agent acts
on a fluent sentence it invented, and downstream the LLM answers a question nobody
asked.

Motivating incidents:
  * a noise burst on an English-only line transcribed as Chinese, and the agent
    replied to it;
  * an English-only agent returned Gujarati script for the phrase "egg roll",
    five times in one call, so the item never reached the order.

Detection here is script-based, not model-based. It is deliberately cheap and
deterministic: if your agent is configured for English and Spanish and the
transcript contains Devanagari, that is a hallucination and no classifier is
needed to say so. Script detection cannot separate languages that share an
alphabet — English from Spanish, Hindi from Marathi — so `allowed_scripts`
answers "is this plausibly one of my languages", not "which language is this".
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Unicode block ranges, keyed by the script name used in `allowed_scripts`.
# Latin covers the ASCII and Latin-1/Extended ranges, which is enough for the
# European languages a phone agent normally serves.
_SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "latin": ((0x0041, 0x024F), (0x1E00, 0x1EFF)),
    "cyrillic": ((0x0400, 0x04FF),),
    "greek": ((0x0370, 0x03FF),),
    "hebrew": ((0x0590, 0x05FF),),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "devanagari": ((0x0900, 0x097F),),
    "bengali": ((0x0980, 0x09FF),),
    "gurmukhi": ((0x0A00, 0x0A7F),),
    "gujarati": ((0x0A80, 0x0AFF),),
    "tamil": ((0x0B80, 0x0BFF),),
    "telugu": ((0x0C00, 0x0C7F),),
    "kannada": ((0x0C80, 0x0CFF),),
    "thai": ((0x0E00, 0x0E7F),),
    "han": ((0x4E00, 0x9FFF), (0x3400, 0x4DBF)),   # Chinese, and kanji in Japanese
    "hiragana": ((0x3040, 0x309F),),
    "katakana": ((0x30A0, 0x30FF),),
    "hangul": ((0xAC00, 0xD7AF), (0x1100, 0x11FF)),
}

# Which scripts a language legitimately produces. Japanese is the case that breaks
# naive checks: it mixes kanji (han) with both kana, so allowing "ja" must allow
# han — which in turn means a ja-enabled agent cannot flag Chinese by script alone.
_LANGUAGE_SCRIPTS: dict[str, tuple[str, ...]] = {
    "en": ("latin",), "es": ("latin",), "fr": ("latin",), "de": ("latin",),
    "it": ("latin",), "pt": ("latin",), "nl": ("latin",), "pl": ("latin",),
    "vi": ("latin",), "id": ("latin",), "tl": ("latin",), "tr": ("latin",),
    "ru": ("cyrillic",), "uk": ("cyrillic",), "el": ("greek",),
    "he": ("hebrew",), "ar": ("arabic",), "fa": ("arabic",),
    "hi": ("devanagari",), "mr": ("devanagari",), "bn": ("bengali",),
    "pa": ("gurmukhi",), "gu": ("gujarati",), "ta": ("tamil",),
    "te": ("telugu",), "kn": ("kannada",), "th": ("thai",),
    "zh": ("han",), "ja": ("han", "hiragana", "katakana"), "ko": ("hangul",),
}


def scripts_for_languages(languages: list[str]) -> set[str]:
    """Scripts implied by a list of ISO 639-1 codes. Unknown codes are ignored
    rather than raising — an unrecognized code should not take down a deploy gate,
    and the caller can pass `allowed_scripts` explicitly if they need precision."""
    allowed: set[str] = set()
    for lang in languages:
        allowed.update(_LANGUAGE_SCRIPTS.get(lang.lower().split("-")[0], ()))
    return allowed


def detect_scripts(text: str) -> set[str]:
    """Scripts present in the text.

    Digits, punctuation and whitespace are ignored — they are script-neutral and
    appear in every language. Characters in no known range are also ignored, so
    an emoji in a transcript does not read as a foreign language.
    """
    found: set[str] = set()
    for ch in text or "":
        if ch.isspace() or ch.isdigit() or not ch.isalnum():
            continue
        code = ord(ch)
        for script, ranges in _SCRIPT_RANGES.items():
            if any(lo <= code <= hi for lo, hi in ranges):
                found.add(script)
                break
    return found


@dataclass
class LanguageCheck:
    ok: bool
    detected: set[str] = field(default_factory=set)
    allowed: set[str] = field(default_factory=set)
    offending: set[str] = field(default_factory=set)
    sample: str = ""

    def __str__(self) -> str:
        if self.ok:
            return f"in-set ({', '.join(sorted(self.detected)) or 'no script chars'})"
        return (
            f"OUT-OF-SET script {sorted(self.offending)} not in {sorted(self.allowed)}"
            f" — {self.sample!r}"
        )


def check_language(
    transcript: str,
    languages: list[str] | None = None,
    allowed_scripts: set[str] | None = None,
) -> LanguageCheck:
    """Flag a transcript containing scripts outside the agent's configured set.

    Pass either `languages` (ISO codes, mapped to scripts) or `allowed_scripts`
    directly. An empty transcript passes: silence is not a hallucination.
    """
    if allowed_scripts is None:
        if languages is None:
            raise ValueError("pass either languages or allowed_scripts")
        allowed_scripts = scripts_for_languages(languages)

    detected = detect_scripts(transcript)
    offending = detected - allowed_scripts
    return LanguageCheck(
        ok=not offending,
        detected=detected,
        allowed=allowed_scripts,
        offending=offending,
        sample=(transcript or "")[:80],
    )


def confidence_gate(confidence: float | None, threshold: float) -> bool:
    """True to keep a final transcript, False to drop it as likely noise.

    Only gates on a *positive* reported confidence. Providers that report nothing
    (None) or a flat zero must always pass, otherwise switching recognizers
    silently drops every turn.

    Set the threshold from measured data rather than intuition, and re-measure
    after changing noise cancellation. A threshold of 0.65 once looked right
    against observed hallucinations, then dropped a caller saying their own name
    at 0.61 — the agent kept re-asking and the caller heard silence. Being too
    strict here is a worse experience than the rare noise turn it prevents.
    """
    if confidence is None or not isinstance(confidence, (int, float)) or confidence <= 0:
        return True
    return threshold <= 0 or confidence >= threshold
