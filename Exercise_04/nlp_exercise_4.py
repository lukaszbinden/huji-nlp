import nltk
from nltk.corpus import brown
from nltk import word_tokenize
from collections import Counter, defaultdict
import re

WORD = 0
TAG = 1
COUNTER_SHOWS = 'counter_shows'
COUNTER_EQUAL = 'counter_equal'
UNKNOWN_TAG = 'NN'
COUNT_CUTOFF = 5
SMOOTH = 1
PSEUDO = 2
PSEUDO_SMOOTH = 3
OTHER = 'other'

# load new category
news_sents = brown.tagged_sents(categories='news')

# divide to sets
set_size = round(len(news_sents) * 0.9)
train_set = news_sents[:set_size]
test_set = news_sents[set_size:]

arr = []
for i in range(len(train_set)):
    sent = train_set[i]
    for word, tag in sent:
        if word not in arr:
            arr.append(word)
train_set_size = len(arr)

PSEUDOWORDS = {
    "\d+.{0,1}\d*$": 'NUM',
    "-year-old$": 'AGE',
    "[$]": 'PRICE',
    "^\d+/\d+/{0,1}\d*$": 'DATE',
    "^\d+-\d+-{0,1}\d*$": 'digitsAndDash',
    "^[A-Z]+$":  'ALLCAPS',
    "^[A-Za-z][.][A-Za-z]([.][A-Za-z])*$": 'INITIALS'
}

def pw(word):
    for pat in PSEUDOWORDS.keys():
        if re.findall(pat, word, re.I):
            return PSEUDOWORDS[pat]
    return word

def eml_add_smooth(yi, xi, eqml):
    """
    Computes emission probability e(x_i | y_i) using maximum likelihood estimation on the training set
    and add-one smoothing.
    :param yi: a label/state y_i
    :param xi: a word x_i
    :param eqml: dictionary for eml where pre-computed values are available
    :return: e(x_i | y_i)
    """
    return (eqml[yi][xi] + 1) / (sum(eqml[yi].values()) + train_set_size)


def eml_use_pseudowords_and_mle(xi, yi, deml):
    """
    Computes emission probability e(x_i | y_i) using maximum likelihood estimation and pseudowords
    on the training set
    :param xi: a word x_i
    :param yi: a label/state y_i
    :param train_set: the training set
    :param deml: dictionary for eml where pre-computed values are available
    :return: q(y_i | y_i-1)
    """
    if xi not in deml[yi]:
        xi = pw(xi) # use pseudo-word instead

    return (deml[yi][xi]) / (sum(deml[yi].values()))

def eml_use_pseudowords_and_smooth(xi, yi, deml):
    """
    Computes emission probability e(x_i | y_i) using maximum likelihood estimation and pseudowords
    on the training set
    :param xi: a word x_i
    :param yi: a label/state y_i
    :param train_set: the training set
    :param deml: dictionary for eml where pre-computed values are available
    :return: q(y_i | y_i-1)
    """
    if xi not in deml[yi]:
        xi = pw(xi) # use pseudo-word instead

    return (deml[yi][xi] + 1) / (sum(deml[yi].values()) + train_set_size)


def compute_propability(word, label, dict):
    """
    Computes probability q(word | label).
    :param word: a word/state y_i
    :param label: a label/state y_i-1
    :param dict: dictionary where pre-computed values are stored
    :return: dict(y_i | y_i-1)
    """
    return dict[label][word] / sum(dict[label].values())

def compute_eml(V_CASE, eqml, k, sent_words, v):
    if V_CASE == PSEUDO:
        eml = eml_use_pseudowords_and_mle(sent_words[k - 1], v, eqml)
    elif V_CASE == SMOOTH:
        eml = eml_add_smooth(v, sent_words[k - 1], eqml)
    elif V_CASE == PSEUDO_SMOOTH:
        eml = eml_use_pseudowords_and_smooth(sent_words[k - 1], v, eqml)
    else:
        eml = compute_propability(sent_words[k - 1], v, eqml)
    return eml


# -----------------------
# ------- (b) -----------
# -----------------------

print(" --------------------- ")
print(" ------- (b) --------- ")
print(" --------------------- ")

def findTags(user_input, tagged_text):
    """
    Get the tag of each word
    :param user_input: the sentence we want to tag
    :param tagged_text: the tags
    :return:
    """
    result = []
    for item in tagged_text:
        for w in user_input:
            if w[WORD] == item[WORD]:
                tup = (w[WORD], item[TAG])
                result.append(tup)
        continue

    return result

def getMostLikelyTag(set_of_sents):
    """
    Get the most likely tags for a set of sentences
    :param set_of_sents: the sentences we want to get the tags for
    :return: dictionary of words and most likley tags
    """
    # initialize tags for the words
    l_of_tags = []
    all_tags = brown.tagged_sents()
    size_of_set = len(set_of_sents)
    for i in range(size_of_set):
        tags = findTags(set_of_sents[i], all_tags[i])
        l_of_tags += tags

    # merge tags for each word
    d = {}  # dict of words and tags amount
    for i in range(len(l_of_tags)):
        w = l_of_tags[i][WORD]
        t = l_of_tags[i][TAG]
        if w in d:
            if t in d[w]:
                d[w][t] = d[w][t] + 1
            else:
                d[w][t] = 1
        else:
            d[w] = {t: 1}

    # get the max tag of each word
    result = {}
    for w, t in d.items():
        v = list(t.values())
        k = list(t.keys())
        fin_tag = k[v.index(max(v))]
        result[w] = fin_tag

    return result


def computeErrorRate(test_set, words_likely_tags):
    """
    Computes all the error rates
    :param test_set: the test set we compare to
    :param words_likely_tags: the set with likely tags
    :return: the error rates
    """
    # initiate vars
    known_words = {}  # those two dictionaries are optional, just for debuging
    unknown_words = {}  # those two dictionaries are optional, just for debuging
    correct_predictions = 0
    total_predictions = 0
    correct_unknown_predictions = 0
    total_unknown_predictions = 0

    for i in range(len(test_set)):  # iterate sentences
        test_sent = test_set[i]
        for j in range(len(test_sent)):  # iterate words in sent
            w = test_sent[j][WORD]
            t = test_sent[j][TAG]

            # known words
            if w in words_likely_tags:
                if w in known_words:
                    known_words[w][COUNTER_SHOWS] += 1
                    if t == words_likely_tags[w]: # same tag
                        known_words[w][COUNTER_EQUAL] += 1
                        correct_predictions += 1
                else:
                    if t == words_likely_tags[w]:  # same tag
                        known_words[w] = {COUNTER_SHOWS: 1, COUNTER_EQUAL: 1}
                        correct_predictions += 1
                    else:
                        known_words[w] = {COUNTER_SHOWS: 1, COUNTER_EQUAL: 0}

                total_predictions += 1
            # unknown words
            else: # w not in words_likely_tags, treat w as unknown_word
                if w in unknown_words:
                    unknown_words[w][COUNTER_SHOWS] += 1
                    if t == UNKNOWN_TAG:
                        # same tag as our model predicts for unknown words
                        unknown_words[w][COUNTER_EQUAL] += 1
                        correct_unknown_predictions += 1
                else:
                    if t == UNKNOWN_TAG:  # same tag
                        unknown_words[w] = {COUNTER_SHOWS: 1, COUNTER_EQUAL: 1}
                        correct_unknown_predictions += 1
                    else:
                        unknown_words[w] = {COUNTER_SHOWS: 1, COUNTER_EQUAL: 0}

                total_unknown_predictions += 1

    # print('correct_predictions......... = ', correct_predictions)
    # print('total_predictions........... = ', total_predictions)
    # print('correct_unknown_predictions. = ', correct_unknown_predictions)
    # print('total_unknown_predictions... = ', total_unknown_predictions)
    err_rate_known = 1 - correct_predictions/total_predictions
    err_rate_unknown = 1 - correct_unknown_predictions/total_unknown_predictions
    # total_err = err_rate_known + err_rate_unknown
    tot_pred = total_predictions + total_unknown_predictions
    corr_pred = correct_predictions + correct_unknown_predictions
    total_err = 1 - corr_pred/tot_pred

    return err_rate_known, err_rate_unknown, total_err


# ANSWER FOR 2)b) (i)
words_likely_tags = getMostLikelyTag(train_set)
# print('words_likely_tags')
# print(words_likely_tags)

# ANSWER FOR 2)b) (ii)
# print('num test sents: ', len(test_set))
err_rate_known, err_rate_unknown, total_err = computeErrorRate(test_set, words_likely_tags)
print('ErrorRates TEST_SET -->')
print(err_rate_known, err_rate_unknown, total_err)
print('ErrorRates TEST_SET  <--')

# -----------------------
# ------- (c) -----------
# -----------------------

print(" --------------------- ")
print(" ------- (c) --------- ")
print(" --------------------- ")

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




################################################
# (c) ii.
# Implement the Viterbi algorithm corresponding to the bigram HMM model in a way you can
# tag any test sentence.
################################################

def viterbi(sent, dqml, eqml, S, V_CASE=-1):
    """
    Executes the Viterbi algorithm given the parameters.
    :param sent: a sentence x1....xn to find the most likely tag sequence y1...yn+1 for
    :param dqml: dictionary for q(s|u) based in training set
    :param deml: dictionary for e(x|s) based in training set
    :param S: set of tags in the training corpus
    :param V_CASE: the algorithm used
    :return: the most likely tag sequence y1...yn+1 for input sent
    """

    if type(sent) is list:
        sent_words = sent
    else:
        sent_words = word_tokenize(sent)
    n = len(sent_words)

    # define and initialize PI table
    pi = defaultdict(Counter)
    pi[0]['*'] = 1
    bp = {}

    for k in range(1, n+1):
        bp[k] = {}
        for v in S:
            eml = compute_eml(V_CASE, eqml, k, sent_words, v)
            if k-1 is 0:  # w e S_0 -> w = '*'
                qmlr = compute_qml(dqml, v, '*')
                pival = pi[0]['*'] * qmlr * eml
                pi[k][v] = pival
                bp[k][v] = '*'
            else:  # for w e S_k, S_k = S
                max_S = None
                max_w = -1
                for w in S:
                    qmlr = compute_qml(dqml, v, w)
                    currmax = pi[k-1][w] * qmlr * eml
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
    yn = None
    for v in S:
        nextmax = pi[n][v] * compute_propability('STOP', v, dqml)
        if nextmax > max_y:
            max_y = nextmax
            yn = v

    # calculate y_n-1....y1
    yk1 = yn
    tagSequence = list()
    tagSequence.append(yn)
    for k in range(n-1,0,-1):
        yk = bp[k+1][yk1]
        tagSequence.append(yk)
        yk1 = yk

    tagSequence.reverse()
    return tagSequence



def compute_qml(dqml, v, w):
    return compute_propability(v, w, dqml)


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


def run_tests_compute_error_rates(dqml, eqml, S, V_CASE=-1, CONF_MATRIX=0):
    test_set_total_tests = 0
    test_set_err_rate_known = 0
    test_set_err_rate_unknown = 0
    test_set_total_err = 0
    if CONF_MATRIX:
        testtags = list()
        predtags = list()
    for test_sent in test_set:
        test_set_total_tests += 1
        sent = [word for word, tag in test_sent]
        tagsequence = viterbi(sent, dqml, eqml, S, V_CASE)
        if CONF_MATRIX:
            predtags.extend(tagsequence)
        err_rate_known, err_rate_unknown, total_err = computeErrorRate(test_sent, tagsequence)
        # print(err_rate_known, err_rate_unknown, total_err)
        test_set_err_rate_known += err_rate_known
        test_set_err_rate_unknown += err_rate_unknown
        test_set_total_err += total_err
        if CONF_MATRIX:
            test_sent_tags = [tag for word, tag in test_sent]
            testtags.extend(test_sent_tags)
    test_set_err_rate_known /= test_set_total_tests
    test_set_err_rate_unknown /= test_set_total_tests
    test_set_total_err /= test_set_total_tests

    if V_CASE == SMOOTH:
        label = 'smoothing | (d) ii'
    elif V_CASE == PSEUDO:
        label = 'pseudo | (e) ii'
    elif V_CASE == PSEUDO_SMOOTH:
        label = 'pseudo+smoothing | (e) iii'
    else:
        label = 'inital | (c) iii'

    print('ErrorRates TEST_SET --> [', label, ']')
    print(test_set_err_rate_known, test_set_err_rate_unknown, test_set_total_err)
    print('ErrorRates TEST_SET  <--')
    if CONF_MATRIX:
        cm = nltk.ConfusionMatrix(testtags, predtags)
        print(cm.pretty_format(sort_by_count=True, show_percents=False))  # , truncate=40


################################################
# (c) (iii)
# Run the algorithm from c)ii) on the test set. Compute the error rate and compare it to the
# results from b)ii).
################################################

dqml = train_qml(train_set)

# compute S (set of tags) from dictionary (only do it once)
S = [tag for tag in dqml]
S.remove('*')

eqml = train_eml(train_set)
run_tests_compute_error_rates(dqml, eqml, S, -1, 0)

# -----------------------
# ------- (d) -----------
# -----------------------

print(" --------------------- ")
print(" ------- (d) --------- ")
print(" --------------------- ")


################################################
# (d) (ii)
# Run the algorithm from c)ii) on the test set. Compute the error rate and compare it to the
# results from b)ii) and c)iii).
################################################

run_tests_compute_error_rates(dqml, eqml, S, SMOOTH, 0)




# -----------------------
# ------- (e) -----------
# -----------------------

print(" --------------------- ")
print(" ------- (e) --------- ")
print(" --------------------- ")

################################################
# (e) i.
# Design a set of pseudo-words for unknown words in the test set and low-frequency words in
# the training set.
# Confer functions pw(tok) and train_eml(train_set) and
# eml_use_pseudowords_and_mle(xi, yi, deml).
################################################





def train_eml_pseudo(train_set):
    """
    Computes emission probabilities e(x_i | y_i) using maximum likelihood estimation on the training set.
    Further, for low-frequency words, replace them with their respective pseudoword.
    :param train_set: the training set
    :return deml: dictionary for e where computed e(y_i | y_i-1) values are stored
    """
    deml = train_eml(train_set)

    # smoothing with pseudo-words:
    # For any word seen in training data less than COUNT_CUTOFF times,
    # we simply replace the word x by its pseudo-word pw(x).
    demlpw = defaultdict(Counter)
    for tag in deml:
        for word in deml[tag]:
            pseudoword = pw(word)

            if deml[tag][word] < COUNT_CUTOFF:
                if OTHER in demlpw[tag]:
                    demlpw[tag][OTHER] += deml[tag][word]
                else:
                    demlpw[tag][OTHER] = deml[tag][word]
            else:
                if pseudoword in demlpw[tag]:
                    demlpw[tag][pseudoword] += deml[tag][word]
                else:
                    demlpw[tag][pseudoword] = deml[tag][word]

    return demlpw




################################################
# (e) (ii)
# Using the pseudo-words as well as maximum likelihood estimation (as in c)i)), run the Viterbi
# algorithm on the test set. Compute the error rate and compare it to the results from b)ii),
# c)iii) and d)ii).
################################################


dqml = train_qml(train_set)
eqml = train_eml_pseudo(train_set)

run_tests_compute_error_rates(dqml, eqml, S, PSEUDO, 0)


################################################
# (e) (iii)
# Using the pseudo-words as well as Add-One smoothing (as in d)i)), run the Viterbi algorithm
# on the test set. Compute the error rate and compare it to the results from b)ii), c)iii), d)ii)
# and e)ii). For the results obtained using both pseudo-words and Add-One smoothing, build a
# confusion matrix and investigate the most frequent errors.
################################################


run_tests_compute_error_rates(dqml, eqml, S, PSEUDO_SMOOTH, 1)
