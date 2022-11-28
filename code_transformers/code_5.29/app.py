from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as grad
codegen_tkn = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
mdl = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

def codegen(intent):
# give input as text which reflects intent of the program.
     #text = " write a function which takes 2 numbers as input and returns the larger of the two"
     input_ids = codegen_tkn(intent, return_tensors="pt").input_ids

     gen_ids = mdl.generate(input_ids, max_length=128)
     response = codegen_tkn.decode(gen_ids[0], skip_special_tokens=True)
     return response

output=grad.Textbox(lines=1, label="Generated Python Code", placeholder="")
inp=grad.Textbox(lines=1, label="Place your intent here")
grad.Interface(codegen, inputs=inp, outputs=output).launch()
