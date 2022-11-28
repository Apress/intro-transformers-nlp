from transformers import T5ForConditionalGeneration, T5Tokenizer
import gradio as grad

text2text_tkn= T5Tokenizer.from_pretrained("t5-small")
mdl = T5ForConditionalGeneration.from_pretrained("t5-small")



def text2text_deductible(sentence1,sentence2):
     inp1 = "rte sentence1: "+sentence1
     inp2 = "sentence2: "+sentence2
     combined_inp=inp1+" "+inp2
     enc = text2text_tkn(combined_inp, return_tensors="pt")
     tokens = mdl.generate(**enc)
     response=text2text_tkn.batch_decode(tokens)
     return response

sent1=grad.Textbox(lines=1, label="Sentence1", placeholder="Text in English")
sent2=grad.Textbox(lines=1, label="Sentence2", placeholder="Text in English")
out=grad.Textbox(lines=1, label="Whether sentence2 is deductible from sentence1")
grad.Interface(text2text_ deductible, inputs=[sent1,sent2], outputs=out).launch()
