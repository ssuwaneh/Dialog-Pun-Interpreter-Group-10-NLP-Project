# dialog_bot.py
import os
import json
import spacy
from google import genai

# Load spaCy model once (speed fix)
nlp = spacy.load("en_core_web_sm")

# Gemini setup
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
MODEL = "gemini-2.5-flash"

def extract_pun_word(sentence):
    """
    Extract a likely pun word from the sentence.
    Currently selects the last noun in the sentence.
    """
    doc = nlp(sentence)
    nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return nouns[-1] if nouns else None

def validate_context(sentence, pun_word, sense_a, sense_b):
    """
    Uses Gemini to determine if both meanings of the pun
    logically fit within the sentence.
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

def analyze_pun(sentence, sense_a="", sense_b=""):
    """
    Main function used by the Gradio interface.
    Automatically extracts pun word and validates context.
    """
    if not sentence:
        return {
            "pun_word": None,
            "sense_a": sense_a,
            "sense_b": sense_b,
            "sense_a_valid": False,
            "sense_b_valid": False,
            "pun_works": False,
            "reason": "No sentence entered."
        }

    pun_word = extract_pun_word(sentence)
    if not pun_word:
        return {
            "pun_word": "None detected",
            "sense_a": sense_a,
            "sense_b": sense_b,
            "sense_a_valid": False,
            "sense_b_valid": False,
            "pun_works": False,
            "reason": "No pun word detected in the sentence."
        }

    result = validate_context(sentence, pun_word, sense_a, sense_b)

    return {
        "pun_word": pun_word,
        "sense_a": sense_a,
        "sense_b": sense_b,
        **result
    }

def build_system_prompt(sentence, analysis):
    """
    Build the prompt used to answer user questions about the pun.
    """
    return f"""You explain puns to people.

Sentence:
"{sentence}"

Pun word:
{analysis['pun_word']}

Meaning A:
{analysis['sense_a']}

Meaning B:
{analysis['sense_b']}

Does the pun work:
{analysis['pun_works']}

Reason:
{analysis['reason']}

Answer questions about this pun conversationally.
"""

def chat(sentence, question, history, analysis):
    """
    Chat function for the Gradio chatbot.
    """
    system_prompt = build_system_prompt(sentence, analysis)
    contents = [
        {"role": "user", "parts": [{"text": system_prompt}]},
        {"role": "model", "parts": [{"text": "Ready for questions about the pun."}]}
    ]
    for user_msg, bot_msg in history:
        contents.append({"role": "user", "parts": [{"text": user_msg}]})
        contents.append({"role": "model", "parts": [{"text": bot_msg}]})
    contents.append({"role": "user", "parts": [{"text": question}]})
    response = client.models.generate_content(model=MODEL, contents=contents)
    return response.text
