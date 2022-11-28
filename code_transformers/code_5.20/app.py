from transformers import AutoModelWithLMHead, AutoTokenizer
import gradio as grad

text2text_tkn = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
mdl = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

def text2text(context,answer):
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = text2text_tkn ([input_text], return_tensors='pt')

    output = mdl.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=64)

    response=text2text_tkn.decode(output[0])    
    return response

context=grad.Textbox(lines=10, label="English", placeholder="Context")
ans=grad.Textbox(lines=1, label="Answer")
out=grad.Textbox(lines=1, label="Genereated Question")
grad.Interface(text2text, inputs=[context,ans], outputs=out).launch()
