import wikipedia, spacy


def extractor_proper_nouns(wikipedia_page):
    """
    Computes ...
    :param wikipedia_page:
    :return:
    """

    nlp_model = spacy.load('en')
    analyzed_page = nlp_model(page)

    result = [w.text for w in analyzed_page if w.pos_ == 'PROPN']

    return result







if __name__ == "__main__":
    print("ex4 -->")


    page = wikipedia.page('Brad Pitt').content
    result = extractor_proper_nouns(page)

    print("result is: ", result)




    print("ex4 <--")
