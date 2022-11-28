from transformers import pipeline
import gradio as grad
mdl_name = "Helsinki-NLP/opus-mt-en-de"
opus_translator = pipeline("translation", model=mdl_name)

def translate(text):
    
    response = opus_translator(text)
    return response
grad.Interface(translate, inputs=["text",], outputs="text").launch()
