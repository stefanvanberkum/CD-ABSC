# Data reader for electronics data (directly to raw data).
#
# https://github.com/stefanvanberkum/CD-ABSC

import nltk

nltk.download('punkt')


def main():
    """
    Reads data for the 2004 electronics datasets.
    """

    in_path = "data/externalData/electronics_reviews_2004/"
    files = ["Apex AD2600 Progressive-scan DVD player.txt", "Camera.txt",
             "Creative Labs Nomad Jukebox Zen Xtra 40GB.txt", "Nokia 6610.txt"]

    for file in files:
        count = 0
        implicit = 0
        conflicts = 0
        other = 0
        name_split = file.split()
        if len(name_split) == 1:
            name = name_split[0].split('.')[0]
        else:
            name = name_split[0]
        out_path = "data/programGeneratedData/BERT/" + name + "/raw_data_" + name + "_2004.txt"
        with open(out_path, "w") as out:
            lines = open(in_path + file).readlines()
            for line in lines[12:]:
                if line[0:3] == "[t]" or line[0:2] == "##":
                    continue
                else:
                    split = line.split("##")
                    line = split[1]
                    line = nltk.word_tokenize(line)
                    line = " ".join(line)
                    aspects = split[0].split(",")
                    aspect_map = map(nltk.word_tokenize, aspects)
                    aspect_words = list(aspect_map)
                    aspect_polarity = []
                    for words in aspect_words:
                        if 'u' in words or 'p' in words:
                            # Remove all implicit targets.
                            implicit += 1
                            continue
                        elif 's' in words or 'cc' in words or 'cs' in words:
                            # Remove all other oddities.
                            other += 1
                            continue
                        aspect = words[0]
                        for word in words[1:]:
                            if word == "-3" or word == "-2" or word == "-1" or word == "+1" or word == "+2" or word == "+3":
                                polarity = int(int(word) / abs(int(word)))
                            elif word == '[' or word == ']':
                                continue
                            else:
                                aspect += " " + word
                        aspect_polarity.append([aspect, polarity])
                    aspect_list = []
                    for a_p in aspect_polarity:
                        aspect_list.append(a_p[0])
                    for a_p in aspect_polarity:
                        if aspect_list.count(a_p[0]) > 1:
                            # Remove conflicts.
                            conflicts += 1
                            continue
                        new_line = line.replace(a_p[0], "$T$")
                        if line == new_line:
                            # Remove implicit targets.
                            implicit += 1
                            continue
                        count += 1
                        out.write(new_line + "\n")
                        out.write(a_p[0] + "\n")
                        out.write(str(a_p[1]) + "\n")
        print("Read " + str(count) + " aspects from " + name)
        print("Implicit: " + str(implicit))
        print("Conflicts: " + str(conflicts))
        print("Other: " + str(other))


if __name__ == '__main__':
    main()
