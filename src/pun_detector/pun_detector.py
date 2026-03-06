import os
import json
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
MODEL = "gemini-2.5-flash"

def detect_pun_word(sentence):
    prompt = f"""
Find the pun word in this sentence.

Sentence:
"{sentence}"

Return STRICT JSON:
{{
    "pun_word": "the word that creates the pun"
}}

If there is no pun return:
{{
    "pun_word": null
}}
"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
    try:
        data = json.loads(text)
        return data["pun_word"]
    except:
        return None
