from transformers import BartForSequenceClassification, BartTokenizer
import gradio as grad
bart_tkn = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
mdl = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')

def classify(text,label): 
    tkn_ids = bart_tkn.encode(text, label, return_tensors='pt')
    tkn_lgts = mdl(tkn_ids)[0]
    entail_contra_tkn_lgts = tkn_lgts[:,[0,2]]
    probab = entail_contra_tkn_lgts.softmax(dim=1)
    response =  probab[:,1].item() * 100 
    return response
txt=grad.Textbox(lines=1, label="English", placeholder="text to be classified")
labels=grad.Textbox(lines=1, label="Label", placeholder="Input a Label")
out=grad.Textbox(lines=1, label="Probablity of label being true is")
grad.Interface(classify, inputs=[txt,labels], outputs=out).launch()
