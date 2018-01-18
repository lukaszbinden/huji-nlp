import wikipedia, spacy


def extractor_proper_nouns(wikipedia_page):
    """
    Computes ...
    :param wikipedia_page:
    :return:
    """

    nlp_model = spacy.load('en')
    analyzed_page = nlp_model(page)

    result = []
    for i in range(len(analyzed_page)):
        if i < len(analyzed_page) - 1:
            w = analyzed_page[i]
            if w.pos_ == 'PROPN':
                word = w.text
                index = 1
                stop = False
                while True:
                    if i + index < len(analyzed_page) - 1:
                        w1 = analyzed_page[i + index]
                        if w1.pos_ == 'PROPN':
                            word = word + ' ' + w1.text
                            index = index + 1
                        else:
                            break
                else:
                    stop = True
                result.append(word)
                if stop:
                    break


    # result = [w.text for w in analyzed_page if w.pos_ == 'PROPN']

    return result







if __name__ == "__main__":
    print("ex4 -->")


    page = wikipedia.page('Brad Pitt').content
    result = extractor_proper_nouns(page)

    print("result is: ", result)




    print("ex4 <--")
