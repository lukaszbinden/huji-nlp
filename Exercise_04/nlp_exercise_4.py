import wikipedia, spacy


def extractor_proper_nouns(document):
    """
    Computes a list of (Subject, Relation, Object) triplets. See comments in code for details.
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
            triplet = (firstQuadruple['tokens'], tokens, secondQuadruple['tokens'])
            pairsTriplets.append(triplet)

    return pairsTriplets

def extractor_dependency_tree(document):
    """

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
        print(properNounHeads)
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
            print('h1=', h1, ', h2=', h2)
            subj = properNounSet[h1]
            obj = properNounSet[h2]
            # condition #1:
            if h1.head == h2.head and h1.dep_ == 'nsubj' and h2.dep_ == 'dobj':
                relation = [h1.head]
                triplet = (subj, relation, obj)
                pairsTriplets.append(triplet)

            # condition #2:
            # TODO at work 


    return pairsTriplets


if __name__ == "__main__":
    print("ex4 -->")

    print("ex4.3a) -->")
    # page = wikipedia.page('Brad Pitt').content
    page = 'John Jerome Smith likes Mary.'
    result = extractor_proper_nouns(page)
    for q in result:
        print(q)
    print("ex4.3a) <--")

    print("ex4.3b) -->")
    # page = wikipedia.page('Brad Pitt').content
    page = 'John Jerome Smith likes Mary.'
    result = extractor_dependency_tree(page)
    for q in result:
        print(q)
    print("ex4.3b) <--")

    print("ex4 <--")
