from transformers import AutoTokenizer, AutoModelWithLMHead
import gradio as grad
text2text_tkn = AutoTokenizer.from_pretrained("deep-learning-analytics/wikihow-t5-small")
mdl = AutoModelWithLMHead.from_pretrained("deep-learning-analytics/wikihow-t5-small")


def text2text_summary(para):
    initial_txt = para.strip().replace("\n","")
    tkn_text = text2text_tkn.encode(initial_txt, return_tensors="pt")

    tkn_ids = mdl.generate(
            tkn_text,
            max_length=250, 
            num_beams=5,
            repetition_penalty=2.5, 
           
            early_stopping=True
        )

    response = text2text_tkn.decode(tkn_ids[0], skip_special_tokens=True)
    return response

para=grad.Textbox(lines=10, label="Paragraph", placeholder="Copy paragraph")
out=grad.Textbox(lines=1, label="Summary")
grad.Interface(text2text_summary, inputs=para, outputs=out).launch()
