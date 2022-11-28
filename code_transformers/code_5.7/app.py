from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import gradio as grad
mdl_name = "google/pegasus-xsum"
pegasus_tkn = PegasusTokenizer.from_pretrained(mdl_name)
mdl = PegasusForConditionalGeneration.from_pretrained(mdl_name)




def summarize(text):
    tokens = pegasus_tkn(text, truncation=True, padding="longest", return_tensors="pt")
    txt_summary = mdl.generate(**tokens)
    response = pegasus_tkn.batch_decode(txt_summary, skip_special_tokens=True)
    return response
txt=grad.Textbox(lines=10, label="English", placeholder="English Text here")
out=grad.Textbox(lines=10, label="Summary")
grad.Interface(summarize, inputs=txt, outputs=out).launch()
