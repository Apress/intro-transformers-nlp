from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoding = tokenizer("This is my first stab at AutoTokenizer")
print(encoding)
