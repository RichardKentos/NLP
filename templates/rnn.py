# based on: https://jkk.name/neural-tagger-tutorial/
import codecs
import sys
import torch
from torch import nn

PAD = "PAD"
DIM_EMBEDDING = 100
LSTM_HIDDEN = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 10
torch.manual_seed(8446)

def read_conll_file(path):
    """
    read in conll file
    
    :param path: path to read from
    :returns: list with sequences of words and labels for each sentence
    """
    data = []
    current_words = []
    current_tags = []

    for line in open(path, encoding='utf-8'):
        line = line.strip()

        if line:
            if line[0] == '#':
                continue # skip comments
            tok = line.split('\t')

            current_words.append(tok[0])
            current_tags.append(tok[1])
        else:
            if current_words:  # skip empty lines
                data.append((current_words, current_tags))
            current_words = []
            current_tags = []

    # check for last one
    if current_tags != []:
        data.append((current_words, current_tags))
    return data

class Vocab():
    def __init__(self, pad_unk):
        """
        A convenience class that can help store a vocabulary
        and retrieve indices for inputs.
        """
        self.pad_unk = pad_unk
        self.word2idx = {self.pad_unk: 0}
        self.idx2word = [self.pad_unk]

    def getIdx(self, word, add=False):
        if word not in self.word2idx:
            if add:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)
            else:
                return self.word2idx[self.pad_unk]
        return self.word2idx[word]

    def getWord(self, idx):
        return self.idx2word(idx)

if len(sys.argv) < 2:
    print('Please provide train data path as an argument')
    exit(1)

train_data = read_conll_file(sys.argv[1])
max_len = max([len(x[0]) for x in train_data ])

# Create vocabularies for both the tokens
# and the tags
token_vocab = Vocab(PAD)
label_vocab = Vocab(PAD)
id_to_token = [PAD]

for tokens, tags in train_data:
    for token in tokens:
        token_vocab.getIdx(token, True)
    for tag in tags:
        label_vocab.getIdx(tag, True)

NWORDS = len(token_vocab.idx2word)
NTAGS = len(label_vocab.idx2word)

# convert text data with labels to indices
def data2feats(inputData, word_vocab, label_vocab):
    feats = torch.zeros((len(inputData), max_len), dtype=torch.long)
    labels = torch.zeros((len(inputData), max_len), dtype=torch.long)
    for sentPos, sent in enumerate(inputData):
        for wordPos, word in enumerate(sent[0][:max_len]):
            wordIdx = token_vocab.getIdx(word)
            feats[sentPos][wordPos] = wordIdx
        for labelPos, label in enumerate(sent[1][:max_len]):
            labelIdx = label_vocab.getIdx(label)
            labels[sentPos][labelPos] = labelIdx
    return feats, labels

train_feats, train_labels = data2feats(train_data, token_vocab, label_vocab)


# convert to batches
num_batches = int(len(train_feats)/BATCH_SIZE)
train_feats_batches = train_feats[:BATCH_SIZE*num_batches].view(num_batches, BATCH_SIZE, max_len)
train_labels_batches = train_labels[:BATCH_SIZE*num_batches].view(num_batches, BATCH_SIZE, max_len)

class TaggerModel(nn.Module):
    def __init__(self, nwords, ntags):
        super().__init__()

        # Create word embeddings
        self.word_embedding = nn.Embedding(nwords, DIM_EMBEDDING)
        # Create input dropout parameter
        self.word_dropout = nn.Dropout(.2)
        # Create LSTM parameters
        self.rnn = nn.RNN(DIM_EMBEDDING, LSTM_HIDDEN, num_layers=1,
                batch_first=True, bidirectional=False)
        # Create output dropout parameter
        self.rnn_output_dropout = nn.Dropout(.3)
        # Create final matrix multiply parameters
        self.hidden_to_tag = nn.Linear(LSTM_HIDDEN, ntags)

    def forward(self, sentences):
        # Look up word vectors
        word_vectors = self.word_embedding(sentences)
        # Apply dropout
        dropped_word_vectors = self.word_dropout(word_vectors)
        rnn_out, _ = self.rnn(dropped_word_vectors, None)
        # Apply dropout
        rnn_out_dropped = self.rnn_output_dropout(rnn_out)
        # Matrix multiply to get scores for each tag
        output_scores = self.hidden_to_tag(rnn_out_dropped)

        # Calculate loss and predictions
        return output_scores

# define the model
model = TaggerModel(NWORDS, NTAGS)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
print('model overview: ')
print(model)
print()

print('epoch   loss      Train acc.')
for epoch in range(EPOCHS):
    model.train() 
    model.zero_grad()

    # Loop over batches
    loss = 0
    match = 0
    total = 0
    for batchIdx in range(0, num_batches):
        output_scores = model.forward(train_feats_batches[batchIdx])
        output_scores = output_scores.view(BATCH_SIZE * max_len, -1)
        flat_labels = train_labels_batches[batchIdx].view(BATCH_SIZE * max_len)
        batch_loss = loss_function(output_scores, flat_labels)

        predicted_labels = torch.argmax(output_scores, 1)
        predicted_labels = predicted_labels.view(BATCH_SIZE, max_len)

        # Run backward pass
        batch_loss.backward()
        optimizer.step()
        model.zero_grad()
        loss += batch_loss.item()
        # Update the number of correct tags and total tags
        for gold_sent, pred_sent in zip(train_labels_batches[batchIdx], predicted_labels):
            for gold_label, pred_label in zip(gold_sent, pred_sent):
                if gold_label != 0:
                    total += 1
                    if gold_label == pred_label:
                        match+= 1
    print('{0: <8}{1: <10}{2}'.format(epoch, '{:.2f}'.format(loss/num_batches), '{:.4f}'.format(match / total)))


def run_eval(feats_batches, labels_batches):
    model.eval()
    match = 0
    total = 0
    for sents, labels in zip(feats_batches, labels_batches):
        output_scores = model.forward(sents)
        predicted_tags  = torch.argmax(output_scores, 2)
        for goldSent, predSent in zip(labels, predicted_tags):
            for goldLabel, predLabel in zip(goldSent, predSent):
                if goldLabel.item() != 0:
                    total += 1
                    if goldLabel.item() == predLabel.item():
                        match+= 1
    return(match/total)

print()
for devPath in sys.argv[2:]:
    BATCH_SIZE=1
    dev_data=read_data(devPath)
    dev_feats, dev_labels = data2feats(dev_data, token_vocab, label_vocab)
    num_batches_dev = int(len(dev_feats)/BATCH_SIZE)

    dev_feats_batches = dev_feats[:BATCH_SIZE*num_batches_dev].view(num_batches_dev, BATCH_SIZE, max_len)
    dev_labels_batches = dev_labels[:BATCH_SIZE*num_batches_dev].view(num_batches_dev, BATCH_SIZE, max_len)
    score = run_eval(dev_feats_batches, dev_labels_batches)
    print('Accuracy ' + devPath + ': {:.4f}'.format(score))

