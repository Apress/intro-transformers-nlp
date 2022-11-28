from transformers import GPT2LMHeadModel,GPT2Tokenizer
import gradio as grad

mdl = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tkn=GPT2Tokenizer.from_pretrained('gpt2')


def generate(starting_text):
    tkn_ids = gpt2_tkn.encode(starting_text, return_tensors = 'pt')
    gpt2_tensors = mdl.generate(tkn_ids,max_length=100,no_repeat_ngram_size=True,num_beams=3)
    response=""
    #response = gpt2_tensors
    for i, x in enumerate(gpt2_tensors):
       response=response+f"{i}: {gpt2_tkn.decode(x, skip_special_tokens=True)}"
    return response




txt=grad.Textbox(lines=1, label="English", placeholder="English Text here")
out=grad.Textbox(lines=1, label="Generated Tensors")
grad.Interface(generate, inputs=txt, outputs=out).launch()


