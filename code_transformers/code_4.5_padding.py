from transformers import AutoTokenizer

bert_tk = AutoTokenizer.from_pretrained("bert-base-cased")
sentences=["This is my first stab at AutoTokenizer","life is what happens when you are planning other things","how are you"]

encoding = bert_tk(sentences,padding=True)
print(encoding)
