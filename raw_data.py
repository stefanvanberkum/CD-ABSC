import os
import re
import xml.etree.ElementTree as ET
from collections import Counter

import nltk

# import en_core_web_sm
# en_nlp = en_core_web_sm.load()
from HAABSA.data_book_hotel import read_book_hotel
from config import *


def main(_):
    book_in = "data/externalData/book_reviews_2019.xml"
    hotel_in = "data/externalData/hotel_reviews_2015.xml"

    book_out = "data/programGeneratedData/BERT/book/raw_data_book_2019.txt"
    hotel_out = "data/programGeneratedData/BERT/hotel/raw_data_hotel_2015.txt"

    with open(book_out, "w") as out:
        out.write("")
    with open(hotel_out, "w") as out:
        out.write("")

    read_book_hotel(book_in, [], {}, [], {}, book_out)
    read_book_hotel(hotel_in, [], {}, [], {}, hotel_out)


def window(iterable, size):  # stack overflow solution for sliding window
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win


def _get_data_tuple(sptoks, asp_termIn, label):
    # Find the ids of aspect term
    aspect_is = []
    asp_term = ' '.join(sp for sp in asp_termIn).lower()
    for _i, group in enumerate(window(sptoks, len(asp_termIn))):
        if asp_term == ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i, _i + len(asp_termIn)))
            break
        elif asp_term in ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i, _i + len(asp_termIn)))
            break

    # print(aspect_is)
    pos_info = []
    for _i, sptok in enumerate(sptoks):
        pos_info.append(min([abs(_i - i) for i in aspect_is]))

    lab = None
    if label == 'negative':
        lab = -1
    elif label == 'neutral':
        lab = 0
    elif label == "positive":
        lab = 1
    else:
        raise ValueError("Unknown label: %s" % lab)

    return pos_info, lab


"""
This function reads data from the xml file

Iput arguments:
@fname: file location
@source_count: list that contains list [<pad>, 0] at the first position [empty input]
and all the unique words with number of occurences as tuples [empty input]
@source_word2idx: dictionary with unique words and unique index [empty input]
.. same for target

Return:
@source_data: list with lists which contain the sentences corresponding to the aspects saved by word indices 
@target_data: list which contains the indices of the target phrases: THIS DOES NOT CORRESPOND TO THE INDICES OF source_data 
@source_loc_data: list with lists which contains the distance from the aspect for every word in the sentence corresponding to the aspect
@target_label: contains the polarity of the aspect (0=negative, 1=neutral, 2=positive)
@max_sen_len: maximum sentence length
@max_target_len: maximum target length

"""


def read_data_2014(fname, source_count, source_word2idx, target_count, target_phrase2idx, file_name):
    if os.path.isfile(fname) == False:
        raise ("[!] Data %s not found" % fname)

    # parse xml file to tree
    tree = ET.parse(fname)
    root = tree.getroot()

    outF = open(file_name, "a")

    # save all words in source_words (includes duplicates)
    # save all aspects in target_words (includes duplicates)
    # finds max sentence length and max targets length
    source_words, target_words, max_sent_len, max_target_len = [], [], 0, 0
    target_phrases = []

    countConfl = 0
    for sentence in root.iter('sentence'):
        sent = sentence.find('text').text
        sentenceNew = re.sub(' +', ' ', sent)
        sptoks = nltk.word_tokenize(sentenceNew)
        for sp in sptoks:
            source_words.extend([''.join(sp).lower()])
        if len(sptoks) > max_sent_len:
            max_sent_len = len(sptoks)
        for aspectTerms in sentence.iter('aspectTerms'):
            for aspectTerm in aspectTerms.findall('aspectTerm'):
                if aspectTerm.get("polarity") == "conflict":
                    countConfl += 1
                    continue
                    # TERM IPV TARGET
                asp = aspectTerm.get('term')
                if asp != 'NULL':
                    aspNew = re.sub(' +', ' ', asp)
                    t_sptoks = nltk.word_tokenize(aspNew)
                    for sp in t_sptoks:
                        target_words.extend([''.join(sp).lower()])
                    target_phrases.append(' '.join(sp for sp in t_sptoks).lower())
                    if len(t_sptoks) > max_target_len:
                        max_target_len = len(t_sptoks)
    if len(source_count) == 0:
        source_count.append(['<pad>', 0])
    source_count.extend(Counter(source_words + target_words).most_common())
    target_count.extend(Counter(target_phrases).most_common())

    for word, _ in source_count:
        if word not in source_word2idx:
            source_word2idx[word] = len(source_word2idx)

    for phrase, _ in target_count:
        if phrase not in target_phrase2idx:
            target_phrase2idx[phrase] = len(target_phrase2idx)

    source_data, source_loc_data, target_data, target_label = list(), list(), list(), list()

    # collect output data (match with source_word2idx) and write to .txt file
    for sentence in root.iter('sentence'):
        sent = sentence.find('text').text
        sentenceNew = re.sub(' +', ' ', sent)
        sptoks = nltk.word_tokenize(sentenceNew)
        if len(sptoks) != 0:
            idx = []
            for sptok in sptoks:
                idx.append(source_word2idx[''.join(sptok).lower()])
            for aspectTerms in sentence.iter('aspectTerms'):
                for aspectTerm in aspectTerms.findall('aspectTerm'):
                    if aspectTerm.get("polarity") == "conflict": continue
                    # TERM IPV TARGET
                    asp = aspectTerm.get('term')
                    if asp != 'NULL':  # removes implicit targets
                        aspNew = re.sub(' +', ' ', asp)
                        t_sptoks = nltk.word_tokenize(aspNew)
                        source_data.append(idx)
                        outputtext = ' '.join(sp for sp in sptoks).lower()
                        outputtarget = ' '.join(sp for sp in t_sptoks).lower()
                        outputtext = outputtext.replace(outputtarget, '$T$')
                        outF.write(outputtext)
                        outF.write("\n")
                        outF.write(outputtarget)
                        outF.write("\n")
                        pos_info, lab = _get_data_tuple(sptoks, t_sptoks, aspectTerm.get('polarity'))
                        pos_info = [(1 - (i / len(idx))) for i in pos_info]
                        source_loc_data.append(pos_info)
                        targetdata = ' '.join(sp for sp in t_sptoks).lower()
                        target_data.append(target_phrase2idx[targetdata])
                        target_label.append(lab)
                        outF.write(str(lab))
                        outF.write("\n")

    outF.close()
    print("Read %s aspects from %s" % (len(source_data), fname))
    print("Conflicts: " + str(countConfl))
    return source_data, source_loc_data, target_data, target_label, max_sent_len, source_loc_data, max_target_len


if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()
