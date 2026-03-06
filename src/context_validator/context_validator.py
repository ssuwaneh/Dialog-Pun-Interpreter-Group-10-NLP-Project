# src/dialog_bot/context_validator.py
import json
from google import genai
import os

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
MODEL = "gemini-2.5-flash"

def validate_context(sentence, pun_word, sense_a, sense_b):
    """
    Validates if both meanings of the pun fit in the sentence.
    Returns a dict with JSON keys:
        sense_a_valid, sense_b_valid, pun_works, reason
    """
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
