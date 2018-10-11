import argparse
import os
import sys

import numpy as np


def read_corpus():
    filename = 't8.shakespeare.txt'
    lines = []
    with open(filename, 'r') as f:
        for pos, line in enumerate(f):
            if 243 < pos < 124440:
                if len(line.strip()) > 0:
                    lines.append(line)
    corpus = " ".join(lines)
    return corpus


def get_charmap(corpus):
    chars = list(set(corpus))
    chars.sort()
    charmap = {c: i for i, c in enumerate(chars)}
    return chars, charmap


def map_corpus(corpus, charmap):
    return np.array([charmap[c] for c in corpus], dtype=np.int64)


def to_text(line, charset):
    return "".join([charset[c] for c in line])


def main(argv):

    # Read and process data
    corpus = read_corpus()
    print("Corpus: {}...{}".format(corpus[:50], corpus[-50:]))
    print("Total character count: {}".format(len(corpus)))
    chars, charmap = get_charmap(corpus)
    charcount = len(chars)
    print("Unique character count: {}".format(len(chars)))
    array = map_corpus(corpus, charmap)


if __name__ == '__main__':
    main(sys.argv[1:])
