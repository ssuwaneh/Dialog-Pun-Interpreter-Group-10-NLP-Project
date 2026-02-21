import os
import json
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
MODEL = "gemini-2.5-flash"


def validate_context(sentence, pun_word, sense_a, sense_b):
    prompt = f"""
You are evaluating whether a pun works.

Sentence: "{sentence}"
Pun word: "{pun_word}"

Sense A: {sense_a}
Sense B: {sense_b}

Return STRICT JSON:

{{
    "sense_a_valid": true or false,
    "sense_b_valid": true or false,
    "pun_works": true or false,
    "reason": "short explanation"
}}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    text = response.text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]

    try:
        return json.loads(text)
    except Exception:
        return {
            "sense_a_valid": False,
            "sense_b_valid": False,
            "pun_works": False,
            "reason": "Could not parse model response."
        }
