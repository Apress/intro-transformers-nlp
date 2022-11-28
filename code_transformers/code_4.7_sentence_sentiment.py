from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
sentences=["This is my first stab at AutoTokenizer","life is what happens when you are planning other things. so plan life accordingly","this is not tasty atall"]

encoding = tokenizer(sentences,padding=True,truncation=True,return_tensors="pt")
print(encoding)

from transformers import AutoModelForSequenceClassification

model_name = "siebert/sentiment-roberta-large-english"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)

pt_outputs = pt_model(**encoding)
print (pt_outputs)

SequenceClassifierOutput(loss=None, logits=tensor([[ 3.0351, -2.1955],
        [-3.6225,  2.7819],
        [ 3.9581, -3.6334]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)

logits=pt_outputs.logits
print (logits)

output = torch.softmax(logits, dim=1).tolist()[1]
print(output)
