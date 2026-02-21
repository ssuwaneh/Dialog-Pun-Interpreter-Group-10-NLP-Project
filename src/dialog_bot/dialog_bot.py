# requires GEMINI_API_KEY environment variable to be set
# get one at https://aistudio.google.com/apikey
import os
from google import genai
from sense_finder import find_senses
from context_validator import validate_context

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
MODEL = "gemini-2.5-flash"


def analyze_pun(sentence):
    senses = find_senses(sentence)
    validation = validate_context(
        sentence, senses["pun_word"], senses["sense_a"], senses["sense_b"]
    )
    return {**senses, **validation}


def build_system_prompt(sentence, analysis):
    return f"""You explain puns to people. Here's one that's been analyzed:

The sentence is: "{sentence}"
The pun word is "{analysis['pun_word']}"
First meaning: {analysis['sense_a']}
Second meaning: {analysis['sense_b']}
First meaning works in context: {analysis['sense_a_valid']}
Second meaning works in context: {analysis['sense_b_valid']}
Does the pun work: {analysis['pun_works']}
Why: {analysis['reason']}

Answer whatever the user asks about this pun. Be conversational, not robotic.
If they ask something off-topic just bring it back to the pun."""


def chat(sentence, question, history, analysis):
    system_prompt = build_system_prompt(sentence, analysis)

    # seed the conversation with the system prompt as a fake user/model exchange
    contents = [
        {"role": "user", "parts": [{"text": system_prompt + "\n\nReady to answer questions."}]},
        {"role": "model", "parts": [{"text": "Got it, I've reviewed the pun analysis. Ask me anything about it."}]},
    ]

    for user_msg, bot_msg in history:
        contents.append({"role": "user", "parts": [{"text": user_msg}]})
        contents.append({"role": "model", "parts": [{"text": bot_msg}]})

    contents.append({"role": "user", "parts": [{"text": question}]})

    response = client.models.generate_content(model=MODEL, contents=contents)
    return response.text
