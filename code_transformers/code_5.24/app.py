from transformers import T5ForConditionalGeneration, T5Tokenizer
import gradio as grad

text2text_tkn= T5Tokenizer.from_pretrained("t5-small")
mdl = T5ForConditionalGeneration.from_pretrained("t5-small")



def text2text_sentiment(text):
     inp = "sst2 sentence: "+text
     enc = text2text_tkn(inp, return_tensors="pt")
     tokens = mdl.generate(**enc)
     response=text2text_tkn.batch_decode(tokens)
     return response

para=grad.Textbox(lines=1, label="English Text", placeholder="Text in English")
out=grad.Textbox(lines=1, label="Sentiment")
grad.Interface(text2text_sentiment, inputs=para, outputs=out).launch()
