from transformers import pipeline
# create a pipeline instance with a tokenizer and model
roberta_pipe = pipeline(
    "sentiment-analysis",
    model="siebert/sentiment-roberta-large-english",
    tokenizer="siebert/sentiment-roberta-large-english",
    return_all_scores = True
)
