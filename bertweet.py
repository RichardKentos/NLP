import nltk
import torch
from transformers import AutoModel, AutoTokenizer 

bertweet = AutoModel.from_pretrained("vinai/bertweet-large")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")

# INPUT TWEET IS ALREADY NORMALIZED!
line = "DHEC confirms HTTPURL via @USER :crying_face:"

# Tokenize the line using NLTK
tokens = nltk.word_tokenize(line)

# Get the POS tags for the tokens
pos_tags = nltk.pos_tag(tokens)

input_ids = torch.tensor([tokenizer.encode(line)])

with torch.no_grad():
    features = bertweet(input_ids)  # Models outputs are now tuples

print("POS tags:", pos_tags)
print("Features:", features)

