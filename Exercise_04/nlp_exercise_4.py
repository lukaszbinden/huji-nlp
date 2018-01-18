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
    index = 1
    for i in range(len(analyzed_page)):
        if index < len(analyzed_page):
            w = analyzed_page[index]
            if w.pos_ == 'PROPN':
                word = w.text
                off = 1
                stop = False
                while True:
                    if index + off < len(analyzed_page) - 1:
                        w1 = analyzed_page[index + off]
                        if w1.pos_ == 'PROPN':
                            word = word + ' ' + w1.text
                            off = off + 1
                        else:
                            break
                    else:
                        stop = True
                        break
                index = index + off  # add at least 1 plus number of consecutive nouns
                result.append(word)
                if stop:
                    break  # stop for loop, reached end of page
            else:
                index = index + 1
        else:
            break  # stop, reached end of page

    return result







if __name__ == "__main__":
    print("ex4 -->")


    page = wikipedia.page('Brad Pitt').content
    result = extractor_proper_nouns(page)

    print("result is: ", result)




    print("ex4 <--")
