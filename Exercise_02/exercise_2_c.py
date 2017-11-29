import nltk
from nltk.corpus import brown
from nltk import word_tokenize
from collections import Counter, defaultdict

# load new categry
news_sents = brown.tagged_sents(categories='news')

# divide to sets
set_size = round(len(news_sents) * 0.9)
train_set = news_sents[:5] # set_size]
test_set = news_sents[set_size:]

print(brown.sents(categories='news')[0])

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

    # compute S (set of tags) from dictionaries
    S = [tag for tag in dqml]
    print(S)
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
        print('k ========================= ', k)
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
                    # print('k, v, w, pi = ', k, v, w, pi)
                    # print(' qml(v,w,dqml) = ',qml(v,w,dqml))
                    # print(' eml(sent_words[k],v,eqml) = ', eml(sent_words[k-1],v,eqml))
                    currmax = pi[k-1][w] * qml(v,w,dqml) * eml(sent_words[k-1],v,eqml)
                    if currmax > max_w:
                        max_w = currmax
                        max_S = w

                pi[k][v] = max_w
                bp[k][v] = max_S
                # print('pi[k] = {v: max_w}')
                # print(k, v, max_w)
                # print('bp[k] = {v: max_S}')
                # print(k, v, max_S)


    # calculate y_n
    max_y = -1
    yn  = None
    for v in S:
        nextmax = pi[n][v] * qml('STOP',v,dqml)
        if nextmax > max_y:
            max_y = nextmax
            yn = v

    print(qml('.','NN-TL',dqml))
    print(eml('.','.',eqml))
    print(yn, max_y)
    print('pi[n][v] = ', pi[n]['AT'])
    print('pi[n][v] = ', pi[n]['NN-TL'])
    print('pi[n][v] = ', pi[n]['.'])
    print('qml(STOP,v,dqml) = ', qml('AT','*',dqml))
    print('qml(STOP,v,dqml) = ', qml('NN-TL','AT',dqml))
    print('qml(STOP,v,dqml) = ', qml('STOP','.',dqml))

    print('yn = ', yn)
    print(pi)
    print(bp)

    # calculate y_n-1....y1
    yk1 = yn;
    tagSequence = list()
    tagSequence.append(yn)
    for k in range(n-1,0,-1):
        print('k: ', k)
        yk = bp[k+1][yk1]
        tagSequence.append(yk)
        yk1 = yk

    tagSequence.reverse()
    return tagSequence


dqml = train_qml(train_set)
eqml = train_eml(train_set)

#mostlikelytagsequence = viterbi('The Fulton County.', dqml, eqml)
sent = brown.sents(categories='news')[0]
mostlikelytagsequence = viterbi(sent, dqml, eqml)
print(mostlikelytagsequence)

# dqml = train_qml(train_set)
# print(dqml['AT'])
# print(dqml['*'])
# print(dqml['*']['AT'])
# print(qml('AT','*',dqml))
#
# sent = 'the ss text from local files and from the web, in order to get hold '
# sent_words = word_tokenize(sent)
# eqml = train_eml(train_set)
# print('---->')
# print(eqml['AT'])
# print(eqml['AT'][sent_words[1]])
# print(sent_words[1])
#
# print(eqml['AT'][sent_words[0]]);
# print(eml(sent_words[0],'AT',eqml))
#
# for k in range(10-1,0,-1):
#     print(k)

# S = [tag for tag in dqml]
# S.remove('*')
# print(S)
# S0 = ['*']
# print(S0)
# for tag in dqml:
#     print('tag: ', tag)
#     print('2: ', dqml[tag])
#
#
# for k in range(1,12):
#     print(k)

