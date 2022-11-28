from transformers import AutoModelForCausalLM, AutoTokenizer,BlenderbotForConditionalGeneration
import torch


chat_tkn = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
mdl = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


#chat_tkn = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
#mdl = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

def converse(user_input, chat_history=[]):
    
    user_input_ids = chat_tkn(user_input + chat_tkn.eos_token, return_tensors='pt').input_ids

    # keep history in the tensor
    bot_input_ids = torch.cat([torch.LongTensor(chat_history), user_input_ids], dim=-1)

    # get response 
    chat_history = mdl.generate(bot_input_ids, max_length=1000, pad_token_id=chat_tkn.eos_token_id).tolist()
    print (chat_history)

    
    response = chat_tkn.decode(chat_history[0]).split("<|endoftext|>")
    
    print("starting to print response")
    print(response)
    
    # html for display
    html = "<div class='mybot'>"
    for x, mesg in enumerate(response):
        if x%2!=0 :
           mesg="Alicia:"+mesg
           clazz="alicia"
        else :
           clazz="user"
        
        
        print("value of x")
        print(x)
        print("message")
        print (mesg)
        
        html += "<div class='mesg {}'> {}</div>".format(clazz, mesg)
    html += "</div>"
    print(html)
    return html, chat_history

import gradio as grad

css = """
.mychat {display:flex;flex-direction:column}
.mesg {padding:5px;margin-bottom:5px;border-radius:5px;width:75%}
.mesg.user {background-color:lightblue;color:white}
.mesg.alicia {background-color:orange;color:white,align-self:self-end}
.footer {display:none !important}
"""
text=grad.inputs.Textbox(placeholder="Lets chat")
grad.Interface(fn=converse,
             theme="default",
             inputs=[text, "state"],
             outputs=["html", "state"],
             css=css).launch()
