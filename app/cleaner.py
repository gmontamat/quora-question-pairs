#!/usr/bin/env python

"""
Data cleaning class
"""

import re
from string import punctuation

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


class DataCleaner(object):

    def __init__(self, df):
        self.df = df
        self.punctuation = punctuation
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer('english')

    def clean_column(self, column_name):
        for i, text in enumerate(self.df[column_name]):
            self.df[column_name][i] = self.clean(text)

    @staticmethod
    def clean(text):
        """Clean the text
        Adapted from the following script:
        https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text/notebook
        """
        text = re.sub(r"\s{2,}", " ", text)  # Remove extra whitespace

        text = re.sub(
            r"(\b)(what|where|how|who|when|why)'s(\b)",
            r"\g<1>\g<2> is\g<3>", text, flags=re.IGNORECASE
        )
        text = re.sub(r"'s(\b)", r"\g<1>", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"can't", "cannot", text, flags=re.IGNORECASE)
        text = re.sub(r"won't", "will not", text, flags=re.IGNORECASE)
        text = re.sub(r"shan't", "shall not", text, flags=re.IGNORECASE)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"(\b)(i'm|im|i m)(\b)", r"\g<1>I am\g<3>", text, flags=re.IGNORECASE)

        # Units of measure
        text = re.sub(r"([0-9]+)( )?(k|K)", r"\g<1>000", text)
        text = re.sub(r"([0-9]+)( )?(km|kms|KMs|KM|Km)", r"\g<1> kilometers", text)

        # Some common acronyms and compound words
        text = re.sub(r"(\b)e.g(\b)", r"\g<1>eg\g<2>", text, flags=re.IGNORECASE)
        text = re.sub(r"(\b)b.g(\b)", r"\g<1>bg\g<2>", text, flags=re.IGNORECASE)
        text = re.sub(r"(\b)9.?11(\b)", r"\g<1>911\g<2>", text)
        text = re.sub(r"(\b)e.?mail(\b)", r"\g<1>email\g<2>", text)
        text = re.sub(r"(\b)dms(\b)", r"\g<1>direct messages\g<2>", text, flags=re.IGNORECASE)
        text = re.sub(r"(\b)cs(\b)", r"\g<1>computer science\g<2>", text, flags=re.IGNORECASE)
        text = re.sub(r"(\b)upvotes(\b)", r"\g<1>up votes\g<2>", text)
        text = re.sub(r"(\b)bestfriend(\b)", r"\g<1>best friend\g<2>", text)
        text = re.sub(r"(\b)ios(\b)", r"\g<1>operating system\g<3>", text, flags=re.IGNORECASE)
        text = re.sub(r"(\b)gps(\b)", r"\g<1>GPS\g<2>", text)
        text = re.sub(r"(\b)gst(\b)", r"\g<1>GST\g<2>", text)
        text = re.sub(r"(\b)dna(\b)", r"\g<1>DNA\g<2>", text)
        text = re.sub(r"(\b)J K(\b)", r"\g<1>JK\g<2>", text)
        text = re.sub(r"(\b)III(\b)", r"\g<1>3\g<2>", text, flags=re.IGNORECASE)
        text = re.sub(r"(\b)iphone(\b)", r"\g<1>phone\g<2>", text, flags=re.IGNORECASE)
        text = re.sub(r"(\b)kms(\b)", r"\g<1>kilometers\g<2>", text, flags=re.IGNORECASE)

        # Some expressions for the same country
        text = re.sub(r"(\b)(the )?usa(\b)", "\g<1>America\g<3>", text)
        text = re.sub(r"(\b)(the )?USA(\b)", "\g<1>America\g<3>", text)
        text = re.sub(r"(\b)(the )?u s(\b)", "\g<1>America\g<3>", text)
        text = re.sub(r"(\b)(the )?US(\b)", "\g<1>America\g<3>", text)
        text = re.sub(r"(\b)the us(\b)", "\g<1>America\g<2>", text)
        text = re.sub(r"(\b)(the )?(uk|Uk|UK)(\b)", "\g<1>United Kingdom\g<4>", text)

        return text

    @staticmethod
    def remove_symbols(text, symbols):
        """Remove symbols from text
        """
        return ''.join([c for c in text if c not in symbols])

    @staticmethod
    def remove_words(text, words):
        """Remove words from text
        """
        return ' '.join([w for w in text.split() if w not in words])

    @staticmethod
    def stem_words(text, stemmer):
        """Shorten words to their stems using a 'stemmer' engine
        """
        return ' '.join([stemmer.stem(word) for word in text.split()])


if __name__ == '__main__':
    from load_data import load_data
    dc = DataCleaner(load_data('../data/train.csv'))
    print dc.clean("What's the best way to start learning robotics?")
