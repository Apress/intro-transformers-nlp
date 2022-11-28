from transformers import AutoTokenizer

bert_base_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
sentences=["This is my first stab at AutoTokenizer","life is what happens when you are planning other things. so plan life accordingly","how are you"]

encoding = bert_base_tokenizer(sentences,padding=True,truncation=True)
print(encoding)
