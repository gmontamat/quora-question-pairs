#!/usr/bin/env python

"""
Test NLTK library features
"""

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


if __name__ == '__main__':
    # Download necessary files to train nltk
    if False:
        nltk.download()

    # Load train dataset
    from load_data import load_data
    train = load_data('../data/train.csv')

    # Stop words
    if False:
        stop_words = set(stopwords.words('english'))
        for q1, q2 in zip(train.head().question1, train.head().question2):
            print ' '.join([
                token for token in nltk.word_tokenize(q1)
                if token.lower() not in stop_words
            ])
            print ' '.join([
                token for token in nltk.word_tokenize(q2)
                if token.lower() not in stop_words
            ])
            print

    # Word stemmer
    if False:
        stemmer = SnowballStemmer('english')
        for q1, q2 in zip(train.head().question1, train.head().question2):
            print ' '.join([stemmer.stem(token) for token in nltk.word_tokenize(q1)])
            print ' '.join([stemmer.stem(token) for token in nltk.word_tokenize(q2)])
            print
