import nltk
from nltk.corpus import brown
from nltk import word_tokenize
from collections import Counter, defaultdict

# load new categry
news_sents = brown.tagged_sents(categories='news')

# divide to sets
set_size = round(len(news_sents) * 0.9)
train_set = news_sents[:set_size] # set_size]
test_set = news_sents[set_size:]

################################################
# (c) i.
# Training phase: Compute the transition and emission probabilities of a bigram HMM tagger
# directly on the training set using maximum likelihood estimation.
# => confer the functions train_eml(), eml(), train_qml(), qml()
################################################

def train_eml(train_set):
    """
    Computes emission probabilities e(x_i | y_i) using maximum likelihood estimation on the training set.
    :param train_set: the training set
    :return deml: dictionary for e where computed e(y_i | y_i-1) values are stored
    """
    size_of_set = len(train_set)
    deml = defaultdict(Counter)
    for i in range(size_of_set):
        sent = train_set[i]
        prior = '*'
        for word, tag in sent:
            deml[tag][word] +=1
            prior = tag
        deml[prior]['STOP'] +=1
    return deml


def eml(xi, yi, deml):
    """
    Computes emission probability e(x_i | y_i) using maximum likelihood estimation on the training set.
    :param xi: a word x_i
    :param yi: a label/state y_i
    :param train_set: the training set
    :param deml: dictionary for eml where pre-computed values are available
    :return: q(y_i | y_i-1)
    """
    return deml[yi][xi] / sum(deml[yi].values())


def train_qml(train_set):
    """
    Computes transition probabilities q(y_i | y_i-1) using maximum likelihood estimation on the given training set.
    :param train_set: the training set
    :return dqml: dictionary for qml where computed q(y_i | y_i-1) values are stored
    """
    size_of_set = len(train_set)
    dqml = defaultdict(Counter)
    for i in range(size_of_set):
        sent = train_set[i]
        prior = '*'
        for word, tag in sent:
            dqml[prior][tag] +=1
            prior = tag
        dqml[prior]['STOP'] +=1
    return dqml

def qml(yi, yi1, dqml):
    """
    Computes transition probability q(y_i | y_i-1) using maximum likelihood estimation on the training set.
    :param yi: a label/state y_i
    :param yi1: a label/state y_i-1
    :param dqml: dictionary for qml where pre-computed values are stored
    :return: qml(y_i | y_i-1)
    """
    return dqml[yi1][yi] / sum(dqml[yi1].values())


################################################
# (c) ii.
# Implement the Viterbi algorithm corresponding to the bigram HMM model in a way you can
# tag any test sentence.
################################################

def viterbi(sent, dqml, eqml):
    """
    Executes the Viterbi algorithm given the parameters.
    :param sent: a sentence x1....xn to find the most likely tag sequence y1...yn+1 for
    :param dqml: dictionary for q(s|u) based in training set
    :param deml: dictionary for e(x|s) based in training set
    :return: the most likely tag sequence y1...yn+1 for input sent
    """
    #print('start viterbi')
    # compute S (set of tags) from dictionaries
    S = [tag for tag in dqml]
    #print(S)
    S.remove('*')

    if type(sent) is list:
        sent_words = sent
    else:
        sent_words = word_tokenize(sent)
    n = len(sent_words)

    # define and initalize PI table
    pi = defaultdict(Counter)
    pi[0]['*'] = 1
    bp = {}

    for k in range(1,n+1):
        #print('k ========================= ', k)
        bp[k] = {}
        for v in S:
            if k-1 is 0: # w e S_0 -> w = '*'
                pival = pi[0]['*'] * qml(v,'*',dqml) * eml(sent_words[0],v,eqml)
                pi[k][v] = pival
                bp[k][v] = '*'
            else: # for w e S_k, S_k = S
                max_S = None
                max_w = -1
                for w in S:
                    currmax = pi[k-1][w] * qml(v,w,dqml) * eml(sent_words[k-1],v,eqml)
                    if currmax > 0 and currmax > max_w:
                        max_w = currmax
                        max_S = w
                # if word is unknown use tag 'NN'
                if max_S is None:
                    max_w = 0.0
                    max_S = 'NN'
                #print('k = ', k, ', max_w = ', max_w, ', max_S = ', max_S, ', v = ', v)
                pi[k][v] = max_w
                bp[k][v] = max_S

    # calculate y_n
    max_y = -1
    yn  = None
    for v in S:
        nextmax = pi[n][v] * qml('STOP',v,dqml)
        if nextmax > max_y:
            max_y = nextmax
            yn = v

    # print('yn = ', yn)
    #print('pi: ', pi)
    #print('bp: ', bp)

    # calculate y_n-1....y1
    yk1 = yn;
    tagSequence = list()
    tagSequence.append(yn)
    for k in range(n-1,0,-1):
        yk = bp[k+1][yk1]
        tagSequence.append(yk)
        yk1 = yk

    tagSequence.reverse()
    return tagSequence


dqml = train_qml(train_set)
eqml = train_eml(train_set)

sent = brown.sents(categories='news')[0]
sent = 'The Fulton County was open for long hours because of the Switzerland alphorn.'
mostlikelytagsequence = viterbi(sent, dqml, eqml)
print('1. most likely tag sequence: ')
print(sent)
print(mostlikelytagsequence)

sent = 'The Huderi Hebedi.'
mostlikelytagsequence = viterbi(sent, dqml, eqml)
print('2. most likely tag sequence: ')
print(sent)
print(mostlikelytagsequence)

for test_sent in test_set:
    sent = [word for word, tag in test_sent]
    mostlikelytagsequence = viterbi(sent, dqml, eqml)
    print('most likely tag sequence for test sentence: ')
    print(sent)
    print(mostlikelytagsequence)


