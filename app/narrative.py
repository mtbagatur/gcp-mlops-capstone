"""Turns numeric attributions into a short business-language summary (requirement G7)."""
import json
import os
import re

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "key_factors": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["summary", "key_factors"],
}

LANGUAGES = {"en": "English", "tr": "Turkish"}

# Rarely, low-latency generation emits stray characters (seen once: "mesleğĿ").
# Anything outside this set triggers one retry with thinking enabled.
_PLAIN_TEXT = re.compile(r"^[A-Za-z0-9\s.,;:%'\"()\-–>+/!?$€&*=—’“”çğıöşüÇĞİÖŞÜâîû]*$")

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(
            vertexai=True,
            project=os.environ["PROJECT_ID"],
            location=os.getenv("GEMINI_LOCATION", "us-central1"),
        )
    return _client


def build_prompt(probability: float, top_factors: list[dict], lang: str = "en") -> str:
    factors = json.dumps(top_factors, ensure_ascii=False)
    return f"""An income model predicted a {probability * 100:.1f}% probability of 'high income' (>50K).
The most influential inputs and their contributions (log-odds; positive raises the probability):
{factors}

Summarise this for a customer service agent in plain {LANGUAGES[lang]}, in at most 3 sentences.
IMPORTANT: these are the model's reasons, NOT causes. Say things like "the model weighed X heavily";
never claim that a feature increases or decreases a person's income.
key_factors: one short phrase per input above, in the same order."""


def is_clean(result: dict) -> bool:
    return bool(_PLAIN_TEXT.match(" ".join([result["summary"], *result["key_factors"]])))


def _generate(prompt: str, thinking_budget: int) -> dict:
    from google.genai import types

    resp = _get_client().models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SCHEMA,
            max_output_tokens=1024,
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        ),
    )
    return json.loads(resp.text)


def narrate(probability: float, top_factors: list[dict], lang: str = "en") -> dict:
    prompt = build_prompt(probability, top_factors, lang)
    result = _generate(prompt, thinking_budget=0)  # ~1.7 s
    if not is_clean(result):
        result = _generate(prompt, thinking_budget=256)  # ~2.4 s, only on a bad answer
    return result
