import nltk
from nltk.corpus import brown

WORD = 0
TAG = 1
COUNTER_SHOWS = 'counter_shows'
COUNTER_EQUAL = 'counter_equal'
UNKNOWN_TAG = 'NN'

# load new categry
news_sents = brown.tagged_sents(categories='news') #brown.sents(categories='news')

# divide to sets
set_size = round(len(news_sents) * 0.9)
train_set = news_sents[:set_size]
test_set = news_sents[set_size:]

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
    #### LUCAS NOTE ###-----**************!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    known_words = {} # those two dictionaries are optional lucas, just for debuging
    unknown_words = {} # those two dictionaries are optional lucas, just for debuging
    #### LUCAS NOTE ###-----**************!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    correct_predictions = 0
    total_predictions = 0
    correct_unknown_predictions = 0
    total_unknown_predictions = 0

    for i in range(len(test_set)): # iterate sentences
        test_sent = test_set[i]
        for j in range(len(test_sent)): # iterate words in sent
            w = test_sent[j][WORD]
            t = test_sent[j][TAG]

            # known words
            if w in words_likely_tags and t != UNKNOWN_TAG:
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
            elif w in words_likely_tags and t == UNKNOWN_TAG:
                if w in unknown_words:
                    unknown_words[w][COUNTER_SHOWS] += 1
                    if t == words_likely_tags[w]: # same tag
                        unknown_words[w][COUNTER_EQUAL] += 1
                        correct_unknown_predictions += 1
                else:
                    if t == words_likely_tags[w]:  # same tag
                        unknown_words[w] = {COUNTER_SHOWS: 1, COUNTER_EQUAL: 1}
                        correct_unknown_predictions += 1
                    else:
                        unknown_words[w] = {COUNTER_SHOWS: 1, COUNTER_EQUAL: 0}

                total_unknown_predictions += 1
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

    err_rate_known = 1 - correct_predictions/total_predictions
    err_rate_unknown = 1 - correct_unknown_predictions/total_unknown_predictions
    # total_err = err_rate_known + err_rate_unknown
    tot_pred = total_predictions + total_unknown_predictions
    corr_pred = correct_predictions + correct_unknown_predictions
    total_err = 1 - corr_pred/tot_pred

    return err_rate_known, err_rate_unknown, total_err

# ANSWER FOR 2)b) (i)
words_likely_tags = getMostLikelyTag(train_set)
print('MostLikelyTags -->')
j = 0
for i in words_likely_tags:
    print(i, words_likely_tags[i])
    j = j + 1
    if j is 20:
        break
print('MostLikelyTags <--')
# ANSWER FOR 2)b) (ii)
err_rate_known, err_rate_unknown, total_err = computeErrorRate(test_set, words_likely_tags)
print('ErrorRates -->')
print(err_rate_known, err_rate_unknown, total_err)
print('ErrorRates <--')



