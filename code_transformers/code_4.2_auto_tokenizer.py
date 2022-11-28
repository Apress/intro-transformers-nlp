from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

encoding = tokenizer("This is my first stab at AutoTokenizer","life is what happens when you are planning other things")
print(encoding)
