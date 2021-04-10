from HAABSA.data_book_hotel import read_book_hotel
from HAABSA.data_rest_lapt import read_rest_lapt


def main():
    # Domain is one of the following: restaurant (2014), laptop (2014), book (2019), hotel (2015).

    domain = "hotel"
    year = 2015

    if domain == "restaurant" or domain == "laptop":
        train_file = "data/externalData/" + domain + "_train_" + str(year) + ".xml"
        test_file = "data/externalData/" + domain + "_test_" + str(year) + ".xml"
        train_out = "data/programGeneratedData/BERT/" + domain + "/raw_data_" + domain + "_train_" + str(year) + ".txt"
        test_out = "data/programGeneratedData/BERT/" + domain + "/raw_data_" + domain + "_test_" + str(year) + ".txt"
        with open(train_out, "w") as out:
            out.write("")
        with open(test_out, "w") as out:
            out.write("")
        read_rest_lapt(in_file=train_file, source_count=[], source_word2idx={}, target_count=[], target_phrase2idx={},
                       out_file=train_out)
        read_rest_lapt(in_file=test_file, source_count=[], source_word2idx={}, target_count=[], target_phrase2idx={},
                       out_file=test_out)
    else:
        in_file = "data/externalData/" + domain + "_reviews_" + str(year) + ".xml"
        out_file = "data/programGeneratedData/BERT/" + domain + "/raw_data_" + domain + "_" + str(year) + ".txt"
        with open(out_file, "w") as out:
            out.write("")
        read_book_hotel(in_file=in_file, source_count=[], source_word2idx={}, target_count=[], target_phrase2idx={},
                        out_file=out_file)


if __name__ == '__main__':
    main()
