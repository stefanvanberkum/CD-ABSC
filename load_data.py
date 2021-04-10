import random

import numpy as np
from sklearn.model_selection import StratifiedKFold

from data_rest_lapt import read_rest_lapt


def load_data_and_embeddings(config, load_data):
    flags = config

    if load_data:
        source_count, target_count = [], []
        source_word2idx, target_phrase2idx = {}, {}

        print('reading training data...')
        train_data = read_rest_lapt(flags.train_data, source_count, source_word2idx, target_count, target_phrase2idx,
                                    flags.train_path)
        print('reading test data...')
        test_data = read_rest_lapt(flags.test_data, source_count, source_word2idx, target_count, target_phrase2idx,
                                   flags.test_path)

        wt = np.random.normal(0, 0.05, [len(source_word2idx), 300])
        count = 0.0
        with open(flags.pretrain_file, 'r', encoding="utf8") as f:
            for line in f:
                content = line.strip().split()
                if content[0] in source_word2idx:
                    wt[source_word2idx[content[0]]] = np.array(list(map(float, content[1:])))
                    count += 1

        print('finished embedding context vectors...')

        # Print data to txt file.
        out_f = open(flags.embedding_path, "w")
        for i, word in enumerate(source_word2idx):
            out_f.write(word)
            out_f.write(" ")
            out_f.write(' '.join(str(w) for w in wt[i]))
            out_f.write("\n")
        out_f.close()
        print((len(source_word2idx) - count) / len(source_word2idx) * 100)

        return len(train_data[0]), len(test_data[0]), train_data[3], test_data[3]

    else:
        # Get statistic properties from txt file.
        train_size, train_polarity_vector = get_stats_from_file(flags.train_path)
        test_size, test_polarity_vector = get_stats_from_file(flags.test_path)

        return train_size, test_size, train_polarity_vector, test_polarity_vector


def get_stats_from_file(path):
    polarity_vector = []
    with open(path, "r") as fd:
        lines = fd.read().splitlines()
        size = len(lines) / 3
        for i in range(0, len(lines), 3):
            # Polarity.
            polarity_vector.append(lines[i + 2].strip().split()[0])
    return size, polarity_vector


def load_hyper_data(config, load_data, percentage=0.8):
    flags = config

    if load_data:
        """Splits a file in 2 given the `percentage` to go in the large file."""
        random.seed(12345)
        with open(flags.train_path, 'r') as fin, \
                open(flags.hyper_train_path, 'w') as f_out_big, \
                open(flags.hyper_eval_path, 'w') as f_out_small:
            lines = fin.readlines()

            chunked = [lines[i:i + 3] for i in range(0, len(lines), 3)]
            random.shuffle(chunked)
            numlines = int(len(chunked) * percentage)
            if numlines % 20 == 1:
                numlines += 1
            for chunk in chunked[:numlines]:
                for line in chunk:
                    f_out_big.write(line)
            for chunk in chunked[numlines:]:
                for line in chunk:
                    f_out_small.write(line)

    # Get statistic properties from txt file.
    train_size, train_polarity_vector = get_stats_from_file(flags.hyper_train_path)
    test_size, test_polarity_vector = get_stats_from_file(flags.hyper_eval_path)

    return train_size, test_size, train_polarity_vector, test_polarity_vector


# NOT USED
def load_cross_validation(config, split_size, load=True):
    flags = config
    if load:
        # words, sent = [], []
        words, sent = [], []

        with open(flags.train_path, encoding='cp1252') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                words.append([lines[i], lines[i + 1], lines[i + 2]])
                sent.append(lines[i + 2].strip().split()[0])
            words = np.asarray(words)

            sent = np.asarray(sent)

            i = 0
            kf = StratifiedKFold(n_splits=split_size, shuffle=True, random_state=12345)
            for train_idx, val_idx in kf.split(words, sent):
                words_1 = words[train_idx]
                words_2 = words[val_idx]
                with open("data/programGeneratedData/crossValidation" + str(flags.year) + '/cross_train_' + str(
                        i) + '.txt', 'w') as train, \
                        open("data/programGeneratedData/crossValidation" + str(flags.year) + '/cross_val_' + str(
                            i) + '.txt', 'w') as val:  # , \
                    for row in words_1:
                        train.write(row[0])
                        train.write(row[1])
                        train.write(row[2])
                    for row in words_2:
                        val.write(row[0])
                        val.write(row[1])
                        val.write(row[2])
                i += 1
        # Get statistic properties from txt file.
    train_size, train_polarity_vector = get_stats_from_file(
        "data/programGeneratedData/crossValidation" + str(flags.year) + '/cross_train_0.txt')
    test_size, test_polarity_vector = [], []
    for i in range(split_size):
        test_size_i, test_polarity_vector_i = get_stats_from_file(
            "data/programGeneratedData/crossValidation" + str(flags.year) + '/cross_val_' + str(i) + '.txt')
        test_size.append(test_size_i)
        test_polarity_vector.append(test_polarity_vector_i)

    return train_size, test_size, train_polarity_vector, test_polarity_vector
