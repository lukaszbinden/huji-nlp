import nltk
import re
from nltk.corpus import brown
from nltk import word_tokenize
from collections import Counter, defaultdict

WORD = 0
TAG = 1
UNKNOWN_TAG = 'NN'
COUNT_CUTOFF = 5

# load new categry
news_sents = brown.tagged_sents(categories='news')

# divide to sets
set_size = round(len(news_sents) * 0.9)
train_set = news_sents[:set_size] # set_size]
test_set = news_sents[set_size:]


################################################
# (e) i.
# Design a set of pseudo-words for unknown words in the test set and low-frequency words in
# the training set.
# Confer functions pw(tok) and train_eml(train_set) and
# eml_use_pseudowords_and_mle(xi, yi, deml).
################################################


PSEUDOWORDS = {"^anti.*": "antifreeze",
               "^de.*": "defrost",
               "^dis.*": "disagree",
               "^(?:en.*|em.*)": "embrace",
               "^fore.*": "forecast",
               "^(?:in.*|im.*)": "infield",
               "^(?:in.*|im.*|il.*|ir.*)": "impossible",
               "^inter.*": "interact",
               "^mid.*": "midway",
               "^mis.*": "misfire",
               "^non.*": "nonsense",
               "^over.*": "overlook",
               "^pre.*": "prefix",
               "^re.*": "return",
               "^semi.*": "semicircle",
               "^sub.*": "submarine",
               "^super.*": "superstar",
               "^trans.*": "transport",
               "^un.*": "unfriendly",
               "^under.*": "undersea"}


def pw2(word):
    for pat in PSEUDOWORDS.keys():
        if re.findall(pat, word, re.I):
            return PSEUDOWORDS[pat]
    return word


def pw(tok):
    """
    Returns the pseudoword of a token.
    :param tok: a word/token
    :return pseudoword: a pseudoword/word class for the given token
    """
    if tok.isdigit():
        return 'ps_Number'
    elif tok.isupper():
        return 'ps_Abrev.'
    elif tok.endswith('ed'):
        return 'ps_-ed'
    elif tok.endswith('ing'):
        return 'ps_-ing'
    elif tok.endswith(("'s", "s'")):
        return 'ps_Gen'
    elif tok.endswith('ed'):
        return 'ps_-ed'
    elif tok.endswith(('ate', 'en', 'ify', 'ise', 'ize')):
        return 'ps_Verb'
    elif tok.endswith(('ly', 'wards', 'wards', 'wise')):
        return 'ps_ADV'
    elif tok.endswith(('age', 'ance', 'ence', 'dom', 'ee', 'er', 'or', 'hood', 'ism', 'ist', 'ty', 'ment', 'ness',
                       'ry', 'ship', 'sion', 'tion', 'xion')):
        return 'ps_Noun'
    elif tok.istitle():
        return 'ps_Noun'
    elif tok.endswith(('able', 'ible', 'al', 'en', 'ese', 'ful', 'i', 'ic', 'ish', 'ive', 'ian', 'less', 'ly', 'ous',
                      'y')):
        return 'ps_ADJ'
    elif '\'' in tok:
        return 'ps_apos'
    elif tok.islower():
        return 'ps_lower'
    elif tok.endswith('%') and tok[:len(tok)-1].isdigit():
        return 'ps_percent'
    else:
        return UNKNOWN_TAG


def train_eml(train_set):
    """
    Computes emission probabilities e(x_i | y_i) using maximum likelihood estimation on the training set.
    Further, for low-frequency words, replace them with their respective pseudoword.
    :param train_set: the training set
    :return deml: dictionary for e where computed e(y_i | y_i-1) values are stored
    """
    size_of_set = len(train_set)
    deml = defaultdict(Counter)
    for i in range(size_of_set):
        sent = train_set[i]
        prior = '*'
        for word, tag in sent:
            deml[tag][word] += 1
            prior = tag
        deml[prior]['STOP'] += 1

    # smoothing with pseudowords:
    # For any word seen in training data less than COUNT_CUTOFF times,
    # we simply replace the word x by its pseudoword pw(x).
    demlpw = defaultdict(Counter)
    for tag in deml:
        for word in deml[tag]:
            if deml[tag][word] < COUNT_CUTOFF:
                pseudoword = pw2(word)
                # print('replace LFW "', word, '" with pw "', pseudoword, '"')
                demlpw[tag][pseudoword] = deml[tag][word]
            else:
                demlpw[tag][word] = deml[tag][word]

    return demlpw


def eml_use_pseudowords_and_addone(xi, yi, deml):
    """
    Computes emission probability e(x_i | y_i) using maximum likelihood estimation and pseudowords
    on the training set
    :param xi: a word x_i
    :param yi: a label/state y_i
    :param train_set: the training set
    :param deml: dictionary for eml where pre-computed values are available
    :return: q(y_i | y_i-1)
    """

    if xi not in eqml[yi]:
        pwr = pw2(xi)  # use pseudoword instead because word xi is unknown
        # print('replace word = ', xi, ' with pw = ', pwr)
        xi = pwr

    return (deml[yi][xi] + 1) / (sum(deml[yi].values()) + len(deml))


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
            dqml[prior][tag] += 1
            prior = tag
        dqml[prior]['STOP'] += 1
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


def viterbi(sent, dqml, eqml, S):
    """
    Executes the Viterbi algorithm given the parameters.
    :param sent: a sentence x1....xn to find the most likely tag sequence y1...yn+1 for
    :param dqml: dictionary for q(s|u) based in training set
    :param deml: dictionary for e(x|s) based in training set
    :param S: S (set of tags) from dictionaries
    :return: the most likely tag sequence y1...yn+1 for input sent
    """
    if type(sent) is list:
        sent_words = sent
    else:
        sent_words = word_tokenize(sent)
    n = len(sent_words)

    # define and initalize PI table
    pi = defaultdict(Counter)
    pi[0]['*'] = 1
    bp = {}

    for k in range(1, n+1):
        bp[k] = {}
        for v in S:
            eml = eml_use_pseudowords_and_addone(sent_words[k-1], v, eqml)
            if k-1 is 0:  # w e S_0 -> w = '*'
                pival = pi[0]['*'] * qml(v, '*', dqml) * eml
                pi[k][v] = pival
                bp[k][v] = '*'
            else: # for w e S_k, S_k = S
                max_S = None
                max_w = -1
                for w in S:
                    currmax = pi[k-1][w] * qml(v, w, dqml) * eml
                    if currmax > 0 and currmax > max_w:
                        max_w = currmax
                        max_S = w
                # if word is unknown use tag 'NN'
                if max_S is None:
                    max_w = 0.0
                    max_S = UNKNOWN_TAG
                pi[k][v] = max_w
                bp[k][v] = max_S

    # calculate y_n
    max_y = -1
    yn  = None
    for v in S:
        nextmax = pi[n][v] * qml('STOP', v, dqml)
        if nextmax > max_y:
            max_y = nextmax
            yn = v

    # calculate y_n-1....y1
    yk1 = yn
    tagSequence = list()
    tagSequence.append(yn)
    for k in range(n-1, 0, -1):
        yk = bp[k+1][yk1]
        tagSequence.append(yk)
        yk1 = yk

    tagSequence.reverse()
    return tagSequence


def computeErrorRate(test_sent, viterbi_tag_sequence):
    """
    Computes all the error rates
    :param test_sent: the test sentence we compare to
    :param viterbi_tag_sequence: the viterbi tag sentence
    :return: the error rates
    """
    # initiate vars
    correct_predictions = 0
    total_predictions = 0
    correct_unknown_predictions = 0
    total_unknown_predictions = 0

    for j in range(len(test_sent)): # iterate tups in sent
        expectedTag = test_sent[j][1]
        actualTag = viterbi_tag_sequence[j]
        if actualTag == UNKNOWN_TAG:
            if expectedTag == UNKNOWN_TAG:
                correct_unknown_predictions += 1
            total_unknown_predictions += 1
        else:
            if actualTag == expectedTag:
                correct_predictions += 1
            total_predictions += 1

    err_rate_known = 1 - correct_predictions/total_predictions
    if total_unknown_predictions == 0:
        err_rate_unknown = 0
    else:
        err_rate_unknown = 1 - correct_unknown_predictions/total_unknown_predictions
    tot_pred = total_predictions + total_unknown_predictions
    corr_pred = correct_predictions + correct_unknown_predictions
    total_err = 1 - corr_pred/tot_pred

    return err_rate_known, err_rate_unknown, total_err


dqml = train_qml(train_set)
eqml = train_eml(train_set)

# compute S (set of tags) from dictionary
S = [tag for tag in dqml]
S.remove('*')

test_set_total_tests = 0
test_set_err_rate_known = 0
test_set_err_rate_unknown = 0
test_set_total_err = 0

################################################
# (e) (ii)
# Using the pseudo-words as well as maximum likelihood estimation (as in c)i)), run the Viterbi
# algorithm on the test set. Compute the error rate and compare it to the results from b)ii),
# c)iii) and d)ii).
################################################
testtags = list()
predtags = list()
i = 0
for test_sent in test_set:
    test_set_total_tests += 1
    sent = [word for word, tag in test_sent]
    tagsequence = viterbi(sent, dqml, eqml, S)
    predtags.extend(tagsequence)
    err_rate_known, err_rate_unknown, total_err = computeErrorRate(test_sent, tagsequence)
    print(err_rate_known, err_rate_unknown, total_err)
    test_set_err_rate_known += err_rate_known
    test_set_err_rate_unknown += err_rate_unknown
    test_set_total_err += total_err
    test_sent_tags = [tag for word, tag in test_sent]
    #print('sent = ', sent)
    #print('tags = ', tagsequence)
    #print('tests = ', test_sent_tags)
    testtags.extend(test_sent_tags)

# print('predtags = ', predtags)
# print('testtags = ', testtags)

test_set_err_rate_known /= test_set_total_tests
test_set_err_rate_unknown /= test_set_total_tests
test_set_total_err /= test_set_total_tests
print('ErrorRates TEST_SET -->')
print(test_set_err_rate_known, test_set_err_rate_unknown, test_set_total_err)
print('ErrorRates TEST_SET  <--')

cm = nltk.ConfusionMatrix(testtags, predtags)
print(cm.pretty_format(sort_by_count=True, show_percents=False))  # , truncate=40


# OUTPUT OF EXECUTION:
# b)ii) most likely tag baseline:
# ErrorRates TEST_SET -->
# Error rate known words, Error rate unkonwn, Total error rate
# 0.08273219116321007 0.7893356643356644 0.16343849840255587
# ErrorRates TEST_SET  <--
#
# c)iii) Viterbi with zero probabilities:
# ErrorRates TEST_SET -->
# 0.6608844311022964 0.7615091607670845 0.7246654387908575
# ErrorRates TEST_SET  <--
#
# d)ii) Viterbi plus add-one smoothing:
# ErrorRates TEST_SET -->
# 0.5663086147489916 0.039646464646464645 0.5538406060998559
# ErrorRates TEST_SET  <--
#
#
# e)ii) Viterbi using pseudo-words and maximum likelihood estimation
# ErrorRates TEST_SET -->
# 0.5999196812603557 0.7348903500082796 0.6895214188625336
# ErrorRates TEST_SET  <--
#
#
# e)iii) Viterbi using pseudo-words and add-one smoothing
# ErrorRates TEST_SET -->
# 0.5583202057962505 0.04098124098124097 0.5460324985805722
# ErrorRates TEST_SET  <--

#################################
# Confusion matrix:
#
# see separate file confusion_matrix.txt



# Conclusion:
# According to the confusion matrix, the algorithm performs poorly with tags FW-NN,, FW-IN, FW-NN-TL, NP-HL, ,-HL.


