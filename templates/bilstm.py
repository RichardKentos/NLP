import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# Hyperparameters
MAX_LEN=32
BATCH_SIZE = 64
LSTM_DIM = 50
EMBED_DIM = 100
NUM_EPOCHS = 5
LEARNING_RATE = 0.01
torch.manual_seed(8446)

def read_topics_data(path):
    text = []
    labels = []
    for lineIdx, line in enumerate(open(path)):
        tok = line.strip().split('\t')
        labels.append(tok[0])
        text.append(tok[1].split(' '))
    return text, labels

topic_train_text, topic_train_labels = read_topics_data('topic-data/train.txt')
topic_dev_text, topic_dev_labels = read_topics_data('topic-data/dev.txt')

PAD = '<PAD>'

label2idx = {PAD:0, 'starwars':1, 'poke': 2, 'muppet': 3}
idx2label = [PAD, 'starwars', 'poke', 'muppet']

word2idx = {PAD:0}
idx2word = [PAD]

# generate word2idxs
for sentPos, sent in enumerate(topic_train_text):
    for wordPos, word in enumerate(sent[:MAX_LEN]):
        if word not in word2idx:
            word2idx[word] = len(idx2word)
            idx2word.append(word)

# function to convert input to labels for use in torch
def data2feats(text2, labels2):
    feats = torch.zeros((len(text2), MAX_LEN), dtype=torch.long)
    labels = torch.zeros((len(text2)), dtype=torch.long)
    for sentPos, sent in enumerate(text2):
        for wordPos, word in enumerate(sent[:MAX_LEN]):
            wordIdx = word2idx[PAD] if word not in word2idx else word2idx[word]
            feats[sentPos][wordPos] = wordIdx
        labels[sentPos] = label2idx[labels2[sentPos]]
    return feats, labels

train_feats, train_labels = data2feats(topic_train_text, topic_train_labels)
dev_feats, dev_labels = data2feats(topic_dev_text, topic_dev_labels)
# feats is a matrix of dataset size by the maximum length (32), filled with word ids (and the PAD id)
# labels is a vector of dataset size, filled with 1,2, and 3's for each topic.

# Simple batching, note that it only uses full batches, if the data is not dividable
# by BATCH_SIZE the remainder is ignored.
num_batches = int(len(train_labels)/BATCH_SIZE)
train_feats_batches = train_feats[:BATCH_SIZE*num_batches].view(num_batches,BATCH_SIZE, MAX_LEN)
train_labels_batches = train_labels[:BATCH_SIZE*num_batches].view(num_batches, BATCH_SIZE)



# Our model consisting of word embeddings, a single bilstm layer, and an output labels
class LangID(nn.Module):
    def __init__(self, embed_dim, lstm_dim, vocab_dim):
        super(LangID, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_dim, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, lstm_dim, bidirectional=True, batch_first=True)
        self.hidden_to_tag = nn.Linear(lstm_dim * 2, len(idx2label))
        self.lstm_dim = lstm_dim
    
    def forward(self, inputs):
        # First encode the input into word representations and run the bilstm
        word_vectors = self.word_embeddings(inputs)
        bilstm_out, _ = self.bilstm(word_vectors)
        #  Now combine (concatenate) the last state of each layer
        backward_out = bilstm_out[:,0,-self.lstm_dim:].squeeze(1)
        forward_out = bilstm_out[:,-1,:self.lstm_dim].squeeze(1)
        bilstm_out = torch.cat((forward_out, backward_out),1)
        # And get the prediction
        y = self.hidden_to_tag(bilstm_out)
        log_probs = F.softmax(y, dim=1)
        return log_probs
    
    def predict(self, inputs):
        # Disable updating the weights
        with torch.no_grad():
            y = self.forward(inputs)
            outputs = torch.argmax(y)
        return outputs


# define the model
langid_model = LangID(EMBED_DIM,LSTM_DIM, len(idx2word))
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(langid_model.parameters(), lr=LEARNING_RATE)
print('model overview: ')
print(langid_model)
print()

print('epoch   loss     total time')
langid_model.train()
start = time.time()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for feats, label in zip(train_feats_batches, train_labels_batches):
        optimizer.zero_grad()
        y = langid_model.forward(feats)
        loss = loss_function(y,label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(str(epoch) +  '       {:.4f}'.format(epoch_loss/len(train_feats_batches)) + '   {:.2f}'.format(time.time() - start))

langid_model.eval()
cor = 0
for instanceIdx in range(len(dev_labels)):
    instanceFeats= dev_feats[instanceIdx]
    label = langid_model.predict(instanceFeats.view(1,32))
    if label == dev_labels[instanceIdx]:
        cor+=1
print()
print('Accuracy: ' + str(cor/len(dev_labels)))



