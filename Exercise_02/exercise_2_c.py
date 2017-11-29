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

def eml(xi, yi, train_set, deml):
    """
    Computes emission probability q(x_i | y_i) using maximum likelihood estimation on the training set.
    :param xi: a word x_i
    :param yi: a label/state y_i
    :param train_set: the training set
    :param deml: dictionary for eml where pre-computed values are available
    :return: q(y_i | y_i-1)
    """

def train_qml(train_set):
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
    :param train_set: the training set
    :param dqml: dictionary for qml where pre-computed values are available
    :return: q(y_i | y_i-1)
    """

    # if yi1 not in dqml:
    #     size_of_set = len(train_set)
    #     for i in range(size_of_set):
    #         sent = train_set[i]
    #         prior = '*'
    #         for word, tag in sent:
    #             dqml[prior][tag] +=1
    #             prior = tag
    #         dqml[prior]['STOP'] +=1

    return dqml[yi1][yi] / sum(dqml[yi1].values())


#dqml = {}  # dict
#dqml = defaultdict(Counter)
dqml = train_qml(train_set)
qnptl = qml('NN', 'AT', dqml)
print('qnptl: ', qnptl)
print('exp: ', 12/19)

qnptl = qml('AT', 'AT', dqml)
print('qnptl: ', qnptl)
print('exp: 0.0')



#for sent in train_set:
#    print(sent)

# print('-------------')
# for i in dqml:
#     print(i, '---', dqml[i])
#     if i == 'AT':
#         print(dqml[i]['NN'])
#         print(sum(dqml[i].values()))
