from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
import gradio as grad
mdl_name = "Helsinki-NLP/opus-mt-en-de"
mdl = AutoModelForSeq2SeqLM.from_pretrained(mdl_name)
my_tkn = AutoTokenizer.from_pretrained(mdl_name)

#opus_translator = pipeline("translation", model=mdl_name)

def translate(text):
    inputs = my_tkn(text, return_tensors="pt")
    trans_output = mdl.generate(**inputs)
    response = my_tkn.decode(trans_output[0], skip_special_tokens=True)

    #response = opus_translator(text)
    return response
grad.Interface(translate, inputs=["text",], outputs="text").launch()
