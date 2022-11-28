from transformers import pipeline
import gradio as grad
zero_shot_classifier = pipeline("zero-shot-classification")


def classify(text,labels):
    classifer_labels = labels.split(",")
    #["software", "politics", "love", "movies", "emergency", "advertisment","sports"]
    response = zero_shot_classifier(text,classifer_labels)
    return response
txt=grad.Textbox(lines=1, label="English", placeholder="text to be classified")
labels=grad.Textbox(lines=1, label="Labels", placeholder="comma separated labels")
out=grad.Textbox(lines=1, label="Classification")
grad.Interface(classify, inputs=[txt,labels], outputs=out).launch()
