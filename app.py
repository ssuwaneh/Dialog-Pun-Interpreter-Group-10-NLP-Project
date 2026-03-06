import gradio as gr
from dialog_bot import analyze_pun, extract_pun_word, validate_context, chat
from pun_detector.detector import is_pun, get_pun_word
import sys
import os

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

current_analysis = {}
current_sentence = ""
current_pun_word = ""
current_sense_a = ""
current_sense_b = ""

EXAMPLE_PUNS = [
    "I used to be a banker but I lost interest",
    "I'm reading a book about anti-gravity it's impossible to put down",
    "The bicycle can't stand on its own because it's two tired",
    "Time flies like an arrow fruit flies like a banana",
]

def to_pairs(history):
    """Convert Gradio chat history dicts into (user, bot) tuples."""
    pairs = []
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
            pairs.append((history[i]["content"], history[i + 1]["content"]))
            i += 2
        else:
            i += 1
    return pairs

def set_pun(sentence):
    """Analyze the pun sentence and display summary."""
    global current_analysis, current_sentence
    global current_pun_word, current_sense_a, current_sense_b

    current_sentence = sentence
    current_pun_word = extract_pun_word(sentence) or "None detected"

    # For now, dummy senses; ideally integrate WordNet or similar
    current_sense_a = "Sense A placeholder"
    current_sense_b = "Sense B placeholder"

    current_analysis = validate_context(sentence, current_pun_word, current_sense_a, current_sense_b)

    summary = (
        f"**Pun word:** {current_pun_word}\n"
        f"**Sense A:** {current_sense_a} (Valid: {current_analysis['sense_a_valid']})\n"
        f"**Sense B:** {current_sense_b} (Valid: {current_analysis['sense_b_valid']})\n"
        f"**Pun works:** {'Yes' if current_analysis['pun_works'] else 'No'}\n"
        f"**Reason:** {current_analysis['reason']}"
    )
    return summary, [], []

def respond(question, history):
    """Generate chatbot response about the pun."""
    if not current_analysis:
        msg = "Please enter a pun sentence first (above) and click Analyze."
        history = history + [{"role": "user", "content": question},
                             {"role": "assistant", "content": msg}]
        return history, history

    pairs = to_pairs(history)
    answer = chat(current_sentence, question, pairs, current_analysis)
    history = history + [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}]
    return history, history

with gr.Blocks(title="Pun Interpreter - Group 10") as demo:
    gr.Markdown("# Pun Interpreter")
    gr.Markdown("Enter a pun below (or pick an example), then ask questions about it.")

    with gr.Row():
        pun_input = gr.Textbox(label="Pun sentence",
                               placeholder="Type a pun here...",
                               scale=4)
        analyze_btn = gr.Button("Analyze", variant="primary", scale=1)

    gr.Examples(examples=EXAMPLE_PUNS, inputs=pun_input, label="Try one of these")

    analysis_display = gr.Markdown(label="Analysis")
    chatbot = gr.Chatbot(height=350, show_label=False)
    state = gr.State([])

    with gr.Row():
        msg_input = gr.Textbox(label="Ask a question about the pun",
                               placeholder="example: Why is this funny?",
                               scale=4)
        send_btn = gr.Button("Send", variant="primary", scale=1)

    # Hook up buttons
    analyze_btn.click(fn=set_pun, inputs=[pun_input],
                      outputs=[analysis_display, state, chatbot])

    send_btn.click(fn=respond, inputs=[msg_input, state],
                   outputs=[chatbot, state]).then(lambda: "", outputs=[msg_input])

    msg_input.submit(fn=respond, inputs=[msg_input, state],
                     outputs=[chatbot, state]).then(lambda: "", outputs=[msg_input])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
