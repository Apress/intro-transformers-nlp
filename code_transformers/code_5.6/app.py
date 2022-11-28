from transformers import AutoModel,AutoTokenizer,AutoModelForSeq2SeqLM
import gradio as grad
mdl_name = "Helsinki-NLP/opus-mt-en-fr"
mdl = AutoModelForSeq2SeqLM.from_pretrained(mdl_name)
my_tkn = AutoTokenizer.from_pretrained(mdl_name)

#opus_translator = pipeline("translation", model=mdl_name)

def translate(text):
    inputs = my_tkn(text, return_tensors="pt")
    trans_output = mdl.generate(**inputs)
    response = my_tkn.decode(trans_output[0], skip_special_tokens=True)

    #response = opus_translator(text)
    return response
txt=grad.Textbox(lines=1, label="English", placeholder="English Text here")
out=grad.Textbox(lines=1, label="French")
grad.Interface(translate, inputs=txt, outputs=out).launch()
