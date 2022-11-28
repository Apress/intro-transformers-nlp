from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import gradio as grad
import ast

mdl_name = "distilbert-base-cased-distilled-squad"
my_pipeline = pipeline('question-answering', model=mdl_name, tokenizer=mdl_name)


def answer_question(question,context):
    text= "{"+"'question': '"+question+"','context': '"+context+"'}"
    
    di=ast.literal_eval(text)
    response = my_pipeline(di)
    return response
grad.Interface(answer_question, inputs=["text","text"], outputs="text").launch() 
