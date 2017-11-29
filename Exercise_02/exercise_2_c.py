import nltk
from nltk.corpus import brown
from collections import Counter, defaultdict

# load new categry
news_sents = brown.tagged_sents(categories='news')

# divide to sets
set_size = round(len(news_sents) * 0.9)
train_set = news_sents[:5] # set_size]
test_set = news_sents[set_size:]

# Training phase: Compute the transition and emission probabilities of a bigram HMM tagger
# directly on the training set using maximum likelihood estimation.

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


#dqml = {}  # dict
#dqml = defaultdict(Counter)
# dqml = train_qml(train_set)
# qnptl = qml('NN', 'AT', dqml)
# print('qnptl: ', qnptl)
# print('exp: ', 12/19)
#
# qnptl = qml('AT', 'AT', dqml)
# print('qnptl: ', qnptl)
# print('exp: 0.0')

deml = train_eml(train_set)
print('-------------')
for i in deml:
    print(i, '---', deml[i])
    if i == 'AT':
        print(deml[i]['The'])
        print(sum(deml[i].values()))
        print(eml('The', i, deml))
        print('exp: ', 4/19)
        print(eml('hebedi', i, deml))

#for sent in train_set:
#    print(sent)

# print('-------------')
# for i in dqml:
#     print(i, '---', dqml[i])
#     if i == 'AT':
#         print(dqml[i]['NN'])
#         print(sum(dqml[i].values()))
