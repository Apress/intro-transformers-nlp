from transformers import pipeline, set_seed
import gradio as grad
gpt2_pipe = pipeline('text-generation', model='distilgpt2')
set_seed(42)

def generate(starting_text):
    response= gpt2_pipe(starting_text, max_length=20, num_return_sequences=5)
    return response
txt=grad.Textbox(lines=1, label="English", placeholder="English Text here")
out=grad.Textbox(lines=1, label="Generated Text")
grad.Interface(generate, inputs=txt, outputs=out).launch()
