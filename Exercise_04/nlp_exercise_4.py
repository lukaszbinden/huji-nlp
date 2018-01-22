import wikipedia, spacy
from random import randint
from datetime import datetime


def extractor_pos(document):
    """
    Computes a list of (Subject, Relation, Object) triplets based only on the POS tags in the document.
    See comments in code for details.
    :param document: the (wikipedia) document to process
    :return: a list of (Subject, Relation, Object) triplets
    """

    nlp_model = spacy.load('en')
    analyzed_page = nlp_model(document)

    result = []
    index = 0
    # 3.a.i) Find all proper nouns in the corpus/document by locating consecutive sequences of
    # tokens with the POS PROPN.
    for i in range(len(analyzed_page)):
        if index < len(analyzed_page):
            w = analyzed_page[index]
            if w.pos_ == 'PROPN':
                quadruple = {'tokens': [], 'text': None, 'index': index, 'length': 1}
                quadruple['tokens'].append(w)
                text = w.text
                off = 1
                stop = False
                while True:
                    if index + off < len(analyzed_page) - 1:
                        w1 = analyzed_page[index + off]
                        if w1.pos_ == 'PROPN':
                            text = text + ' ' + w1.text
                            quadruple['tokens'].append(w1)
                            quadruple['length'] = quadruple['length'] + 1
                            off = off + 1
                        else:
                            break
                    else:
                        stop = True
                        break
                index = index + off  # add at least 1 plus number of consecutive nouns
                quadruple['text'] = text
                result.append(quadruple)
                if stop:
                    break  # stop for loop, reached end of page
            else:
                index = index    + 1
        else:
            break  # stop, reached end of page

    # 3.a.ii) Find all pairs of proper nouns such that all the tokens between them are
    # non-punctuation (do not have the POS tag PUNCT) and at least one of the tokens between
    # them is a verb (has the POS VERB).
    pairsTriplets = []
    for i in range(len(result) - 1):
        firstQuadruple = result[i]
        startIndex = firstQuadruple['index'] + firstQuadruple['length']
        secondQuadruple = result[i+1]
        endIndex = secondQuadruple['index']

        tokens = []
        noPunct = True
        hasVerb = False
        for j in range(startIndex, endIndex):
            w = analyzed_page[j]
            if w.pos_ == 'PUNCT':
                noPunct = False
                break  # no use in continuing loop
            if w.pos_ == 'VERB':
                hasVerb = True
            if w.pos_ == 'VERB' or w.pos_ == 'ADP':
                tokens.append(w)

        if noPunct and hasVerb:
            # Upon detecting a pair of proper nouns as above, output a (Subject, Relation, Object) triplet
            # get corresponding sentence to print later..
            sent = get_sent(tokens, analyzed_page)
            triplet = [(firstQuadruple['tokens'], tokens, secondQuadruple['tokens']), sent]
            pairsTriplets.append(triplet)

    return pairsTriplets


def get_sent(tokens, analyzed_page):
    indexInDoc = tokens[0].i
    # (this solution is not very good but it does the job)
    for next in list(analyzed_page.sents):
        if indexInDoc >= next[0].i and indexInDoc <= next[len(next)-1].i:
            sent = next
            break

    assert('sent' in locals())
    return sent


def extractor_dependency_tree(document):
    """
    Computes a list of (Subject, Relation, Object) triplets based on the dependency tree
    in the document. See comments in code for details.
    :param document: the (wikipedia) document to process
    :return: a list of (Subject, Relation, Object) triplets
    """
    nlp_model = spacy.load('en')
    analyzed_page = nlp_model(document)
    sents = list(analyzed_page.sents)

    pairsTriplets = []
    for sent in sents:
        properNounHeads = []
        for t in sent:
            if t.pos_ == 'PROPN' and t.dep_ != 'compound':
                properNounHeads.append(t)
        # print(properNounHeads)
        properNounSet = {}
        for head in properNounHeads:
            properNounSet[head] = []
            for t in sent:
                if t.head == head and t.dep_ == 'compound':
                    properNounSet[head].append(t)
            properNounSet[head].append(head)
        # for head in properNounSet:
        #     print(properNounSet[head])
        keys = list(properNounSet.keys())
        for i in range(len(keys) - 1):
            h1 = keys[i]
            h2 = keys[i + 1]
            # print('h1=', h1, ', h2=', h2)
            subj = properNounSet[h1]
            obj = properNounSet[h2]
            # condition #1:
            if h1.head == h2.head and h1.dep_ == 'nsubj' and h2.dep_ == 'dobj':
                relation = [h1.head]
                triplet = [(subj, relation, obj), sent]
                pairsTriplets.append(triplet)
                continue

            # condition #2:
            if h1.head == h2.head.head and h1.dep_ == 'nsubj' and h2.head.dep_ == 'prep' and h2.dep_ == 'pobj':
                relation = [h1.head, h2.head]
                triplet = [(subj, relation, obj), sent]
                pairsTriplets.append(triplet)
                continue
    return pairsTriplets


def run_extractors(page):
    print("run_extractors -->")
    print("input: ", page)
    [print(t) for t in extractor_pos(page)]
    [print(t) for t in extractor_dependency_tree(page)]
    print("run_extractors <--")


def print_to_file(f, fsents, title, posResult, dtResult):
    print(title, file=f)
    print(title, file=fsents)
    print("********** POS_extractor 30 random triplets..............: ", file=f)
    print("********** POS_extractor 30 random triplets..............: ", file=fsents)
    randIndices = set()
    while len(randIndices) < 30 and len(randIndices) != len(posResult):
        randIndices.add(randint(0, len(posResult) - 1))
    id = 1
    for index in randIndices:
        print_line(posResult[index], id, f, fsents)
        id = id + 1
    print("\n********** Dependency_tree__extractor 30 random triplets..: ", file=f)
    print("\n********** Dependency_tree__extractor 30 random triplets..: ", file=fsents)
    randIndices.clear()
    while len(randIndices) < 30 and len(randIndices) != len(dtResult):
        randIndices.add(randint(0, len(dtResult) - 1))
    id = 1
    for index in randIndices:
        print_line(dtResult[index], id, f, fsents)
        id = id + 1
    print("\n\n", file=f)


def print_line(triplet, id, f, fs):
    idStr = str(id) + ":"
    print(idStr, triplet[0], file=f)
    print(idStr, triplet[1], file=fs)


if __name__ == "__main__":
    print("ex4 -->")

    # print("3a) 3b) -->")
    # page = 'John Jerome Smith likes Mary.'
    # run_extractors(page)
    # page = 'John Jerome Smith met with Mary.'
    # run_extractors(page)
    # print("3a) 3b) <--")

    print("3c) -->")

    f = open("ex4_random_triplets.txt", "w")
    f_sents = open("ex4_random_triplets_sents.txt", "w")
    now = datetime.now()
    print("Run at ", now, "\n", file=f)
    print("Run at ", now, "\n", file=f_sents)
    print("Run at ", now)
    trump = wikipedia.page('Donald Trump').content
    print("Donald Trump page:")
    trumpPosResult = extractor_pos(trump)
    print("POS_extractor #triplets..............: ", len(trumpPosResult))
    trumpDtResult = extractor_dependency_tree(trump)
    print("Dependency_tree_extractor #triplets..: ", len(trumpDtResult))
    print_to_file(f, f_sents, "********** Donald Trump page:", trumpPosResult, trumpDtResult)

    bradPitt = wikipedia.page('Brad Pitt').content
    print("Brad Pitt page:")
    bradPittPosResult = extractor_pos(bradPitt)
    print("POS_extractor #triplets..............: ", len(bradPittPosResult))
    bradPittDtResult = extractor_dependency_tree(bradPitt)
    print("Dependency_tree_extractor #triplets..: ", len(bradPittDtResult))
    print_to_file(f, f_sents, "********** Brad Pitt page:", bradPittPosResult, bradPittDtResult)

    jolie = wikipedia.page('Angelina Jolie').content
    print("Angelina Jolie page:")
    joliePosResult = extractor_pos(jolie)
    print("POS_extractor #triplets..............: ", len(joliePosResult))
    jolieDtResult = extractor_dependency_tree(jolie)
    print("Dependency_tree_extractor #triplets..: ", len(jolieDtResult))
    print_to_file(f, f_sents, "********** Angelina Jolie page:", joliePosResult, jolieDtResult)
    f.close()

    print("3c) <--")

    print("ex4 <--")
