import sys

def readBIO(path):
    ents = []
    curEnts = []
    for line in open(path):
        line = line.strip()
        if line == '':
            ents.append(curEnts)
            curEnts = []
        elif line[0] == '#' and len(line.split('\t')) == 1:
            continue
        else:
            curEnts.append(line.split('\t')[2])
    return ents

def toSpans(tags):
    spans = set()
    for beg in range(len(tags)):
        if tags[beg][0] == 'B':
            end = beg
            for end in range(beg+1, len(tags)):
                if tags[end][0] != 'I':
                    break
            spans.add(str(beg) + '-' + str(end) + ':' + tags[beg][2:])
    return spans

def getInstanceScores(predPath, goldPath):
    goldEnts = readBIO(goldPath)
    predEnts = readBIO(predPath)
    entScores = []
    tp = 0
    fp = 0
    fn = 0
    for goldEnt, predEnt in zip(goldEnts, predEnts):
        sentenceLength = len(goldEnt)
        predEnt = predEnt[:sentenceLength]
        goldSpans = toSpans(goldEnt)
        predSpans = toSpans(predEnt)
        overlap = len(goldSpans.intersection(predSpans))
        tp += overlap
        fp += len(predSpans) - overlap
        fn += len(goldSpans) - overlap
        
    prec = 0.0 if tp+fp == 0 else tp/(tp+fp)
    rec = 0.0 if tp+fn == 0 else tp/(tp+fn)
    f1 = 0.0 if prec+rec == 0.0 else 2 * (prec * rec) / (prec + rec)
    return f1
    
    

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Please provide path to gold file and output of your system (in tab-separated format)')
        print('For example: \npython3 eval.py opener_en-dev.conll bert_out-dev.conll')
        print()
        print('The files should be in a conll-like (i.e. tab-separated) format. Each word should be on a separate line, and line breaks are indicated with an empty line. The script will assume to find the BIO annotations on the 3th column. So: index<TAB>word<TAB>label.')

    else:
        score = getInstanceScores(sys.argv[1], sys.argv[2])
        print(score)
