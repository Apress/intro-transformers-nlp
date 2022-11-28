from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer

mdl = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
brt_tkn = BertTokenizer.from_pretrained('bert-base-cased')

sentenceA = 'John went to the restaurant'

sentenceB = 'The kite is flying high in the sky'

encoded = brt_tkn.encode_plus(sentenceA, text_pair=sentenceB, return_tensors='pt')

sentence_relationship_logits = mdl(**encoded)[0]

probablities = softmax(sentence_relationship_logits, dim=1)

print(probablities)
