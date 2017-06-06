#!/usr/bin/env python

"""
Generate features from data
"""

import gensim
import fuzzywuzzy.fuzz as fuzz
import numpy as np
import pandas as pd

from collections import Counter
from functools import partial
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis


class FeatureCreator(object):

    def __init__(self, df, q1_column='question1', q2_column='question2'):
        self.df = df
        self.q1_column = q1_column
        self.q2_column = q2_column
        self.stop_words = set(stopwords.words('english'))
        self.w2c_model = None

    def add_basic_features(self):
        self.df['len_q1'] = self.df[self.q1_column].apply(lambda x: len(str(x)))
        self.df['len_q2'] = self.df[self.q2_column].apply(lambda x: len(str(x)))
        self.df['diff_len'] = self.df.len_q1 - self.df.len_q2
        self.df['len_char_q1'] = self.df[self.q1_column].apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        self.df['len_char_q2'] = self.df[self.q2_column].apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        self.df['len_word_q1'] = self.df[self.q1_column].apply(lambda x: len(str(x).split()))
        self.df['len_word_q2'] = self.df[self.q2_column].apply(lambda x: len(str(x).split()))
        self.df['common_words'] = self.df.apply(
            lambda x: len(
                set(str(x[self.q1_column]).lower().split()).intersection(set(str(x[self.q2_column]).lower().split()))
            ), axis=1
        )

    def add_additional_features(self):
        """Compute TF-IDF and some other interesting features
        """
        # Create columns with list of words for each question
        self.df['q1_words'] = self.df[self.q1_column].map(lambda x: str(x).lower().split())
        self.df['q2_words'] = self.df[self.q2_column].map(lambda x: str(x).lower().split())
        # Compute word frequency and weights for TF-IDF
        questions = pd.Series(self.df['q1_words'].tolist() + self.df['q2_words'].tolist())
        words = [word for question in questions for word in question]
        word_count = Counter(words)
        weights = {word: self.compute_weight(count) for word, count in word_count.iteritems()}
        # Add features
        self.df['word_match'] = self.df.apply(partial(self.word_match_share), axis=1, raw=True)
        self.df['tfidf_word_match'] = self.df.apply(
            partial(self.tfidf_word_match_share, weights=weights, ignore_stop_words=True), axis=1, raw=True
        )
        self.df['tfidf_word_match_stops'] = self.df.apply(
            partial(self.tfidf_word_match_share, weights=weights, ignore_stop_words=False), axis=1, raw=True
        )
        self.df['jaccard_similarity'] = self.df.apply(partial(self.jaccard_similarity), axis=1, raw=True)
        self.df['word_count_diff'] = self.df.apply(partial(self.word_count_diff), axis=1, raw=True)
        self.df['word_count_ratio'] = self.df.apply(partial(self.word_count_ratio), axis=1, raw=True)
        self.df['unique_word_count_diff'] = self.df.apply(partial(self.unique_word_count_diff), axis=1, raw=True)
        self.df['unique_word_count_ratio'] = self.df.apply(partial(self.unique_word_count_ratio), axis=1, raw=True)
        self.df['unique_nonstop_word_count_diff'] = self.df.apply(
            partial(self.unique_nonstop_word_count_diff), axis=1, raw=True
        )
        self.df['unique_nonstop_word_count_ratio'] = self.df.apply(
            partial(self.unique_nonstop_word_count_ratio), axis=1, raw=True
        )
        self.df['same_start'] = self.df.apply(partial(self.same_start), axis=1, raw=True)
        self.df['char_diff'] = self.df.apply(partial(self.char_diff), axis=1, raw=True)
        self.df['char_diff_unique_nonstop'] = self.df.apply(partial(self.char_diff_unique_nonstop), axis=1, raw=True)
        self.df['total_unique_words'] = self.df.apply(partial(self.total_unique_words), axis=1, raw=True)
        self.df['total_unique_words_nonstop'] = self.df.apply(
            partial(self.total_unique_words_nonstop), axis=1, raw=True
        )
        self.df['char_ratio'] = self.df.apply(partial(self.char_ratio), axis=1, raw=True)
        # Remove columns used for calculations
        self.df = self.df.drop(['q1_words', 'q2_words'], axis=1)

    @staticmethod
    def compute_weight(count, epsilon=10000, min_count=2):
        if count < min_count:
            return .0
        else:
            return 1.0 / (count + epsilon)

    def word_match_share(self, row):
        q1_words = set()
        q2_words = set()
        for word in row['q1_words']:
            if word not in self.stop_words:
                q1_words.add(word)
        for word in row['q2_words']:
            if word not in self.stop_words:
                q2_words.add(word)
        if not q1_words or not q2_words:
            return .0
        shared_words_q1 = np.sum([1.0 for word in q1_words if word in q2_words])
        shared_words_q2 = np.sum([1.0 for word in q2_words if word in q1_words])
        return (shared_words_q1 + shared_words_q2) / (len(q1_words) + len(q2_words))

    def tfidf_word_match_share(self, row, weights, ignore_stop_words):
        q1_words = set()
        q2_words = set()
        if ignore_stop_words:
            for word in row['q1_words']:
                q1_words.add(word)
            for word in row['q2_words']:
                q2_words.add(word)
        else:
            for word in row['q1_words']:
                if word not in self.stop_words:
                    q1_words.add(word)
            for word in row['q2_words']:
                if word not in self.stop_words:
                    q2_words.add(word)
        if not q1_words or not q2_words:
            return .0
        shared_weights = np.sum([weights.get(word, .0) for word in q1_words if word in q2_words])
        shared_weights += np.sum([weights.get(word, .0) for word in q2_words if word in q1_words])
        total_weights = np.sum([weights.get(word, .0) for word in q1_words])
        total_weights += np.sum([weights.get(word, .0) for word in q2_words])
        return shared_weights / total_weights

    @staticmethod
    def jaccard_similarity(row):
        words_in_common = set(row['q1_words']).intersection(set(row['q2_words']))
        unique_words = set(row['q1_words']).union(row['q2_words'])
        if not unique_words:
            return 1.0
        return len(words_in_common) / float(len(unique_words))

    @staticmethod
    def word_count_diff(row):
        return abs(len(row['q1_words']) - len(row['q1_words']))

    @staticmethod
    def word_count_ratio(row):
        l1 = float(len(row['q1_words']))
        l2 = len(row['q2_words'])
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    @staticmethod
    def unique_word_count_diff(row):
        return abs(len(set(row['q1_words'])) - len(set(row['q2_words'])))

    @staticmethod
    def unique_word_count_ratio(row):
        l1 = float(len(set(row['q1_words'])))
        l2 = len(set(row['q2_words']))
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    def unique_nonstop_word_count_diff(self, row):
        return abs(
            len([word for word in set(row['q1_words']) if word not in self.stop_words]) -
            len([word for word in set(row['q2_words']) if word not in self.stop_words])
        )

    def unique_nonstop_word_count_ratio(self, row):
        l1 = float(len([word for word in set(row['q1_words']) if word not in self.stop_words]))
        l2 = len([word for word in set(row['q2_words']) if word not in self.stop_words])
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    @staticmethod
    def same_start(row):
        if not row['q1_words'] or not row['q2_words']:
            return np.nan
        return int(row['q1_words'][0] == row['q2_words'][0])

    @staticmethod
    def char_diff(row):
        return abs(len(''.join(row['q1_words'])) - len(''.join(row['q2_words'])))

    def char_diff_unique_nonstop(self, row):
        return abs(
            len(''.join([word for word in set(row['q1_words']) if word not in self.stop_words])) -
            len(''.join([word for word in set(row['q1_words']) if word not in self.stop_words]))
        )

    @staticmethod
    def total_unique_words(row):
        return len(set(row['q1_words']).union(row['q2_words']))

    def total_unique_words_nonstop(self, row):
        return len([word for word in set(row['q1_words']).union(row['q2_words']) if word not in self.stop_words])

    @staticmethod
    def char_ratio(row):
        l1 = len(''.join(row['q1_words']))
        l2 = len(''.join(row['q2_words']))
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    def add_fuzz_features(self):
        self.df['fuzz_qratio'] = self.df.apply(
            lambda x: fuzz.QRatio(str(x[self.q1_column]), str(x[self.q2_column])), axis=1
        )
        self.df['fuzz_wratio'] = self.df.apply(
            lambda x: fuzz.WRatio(str(x[self.q1_column]), str(x[self.q2_column])), axis=1)
        self.df['fuzz_partial_ratio'] = self.df.apply(
            lambda x: fuzz.partial_ratio(str(x[self.q1_column]), str(x[self.q2_column])), axis=1
        )
        self.df['fuzz_partial_token_set_ratio'] = self.df.apply(
            lambda x: fuzz.partial_token_set_ratio(str(x[self.q1_column]), str(x[self.q2_column])), axis=1
        )
        self.df['fuzz_partial_token_sort_ratio'] = self.df.apply(
            lambda x: fuzz.partial_token_sort_ratio(str(x[self.q1_column]), str(x[self.q2_column])), axis=1
        )
        self.df['fuzz_token_set_ratio'] = self.df.apply(
            lambda x: fuzz.token_set_ratio(str(x[self.q1_column]), str(x[self.q2_column])), axis=1
        )
        self.df['fuzz_token_sort_ratio'] = self.df.apply(
            lambda x: fuzz.token_sort_ratio(str(x[self.q1_column]), str(x[self.q2_column])), axis=1
        )

    def add_word2vec_features(self, model_path, model_name='w2v', vector_size=300):
        """ word2vec features require a lot of RAM to be computed
        """
        # Load model and compute Word Mover's Distance
        self.w2c_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.w2c_model.init_sims(replace=True)
        self.df['{}_norm_wmd'.format(model_name)] = self.df.apply(
            lambda x: self.word_mover_distance(x['question1'], x['question2']), axis=1
        )
        self.w2c_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.df['{}_wmd'.format(model_name)] = self.df.apply(
            lambda x: self.word_mover_distance(x['question1'], x['question2']), axis=1
        )
        # Generate vectors from questions
        question1_vectors = np.zeros((self.df.shape[0], vector_size))
        question2_vectors = np.zeros((self.df.shape[0], vector_size))
        j = 0
        for i, row in self.df.iterrows():
            question1_vectors[j, :] = self.text2vec(row[self.q1_column])
            question2_vectors[j, :] = self.text2vec(row[self.q2_column])
            j += 1
        self.w2c_model = None  # Save up some RAM
        # Compute several features using vectors
        self.df['{}_cosine_distance'.format(model_name)] = [
            cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))
        ]
        self.df['{}_cityblock_distance'.format(model_name)] = [
            cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))
        ]
        self.df['{}_jaccard_distance'.format(model_name)] = [
            jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))
        ]
        self.df['{}_canberra_distance'.format(model_name)] = [
            canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))
        ]
        self.df['{}_euclidean_distance'.format(model_name)] = [
            euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))
        ]
        self.df['{}_minkowski_distance'.format(model_name)] = [
            minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))
        ]
        self.df['{}_braycurtis_distance'.format(model_name)] = [
            braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))
        ]
        self.df['{}_skew_q1vec'.format(model_name)] = [skew(x) for x in np.nan_to_num(question1_vectors)]
        self.df['{}_skew_q2vec'.format(model_name)] = [skew(x) for x in np.nan_to_num(question2_vectors)]
        self.df['{}_kur_q1vec'.format(model_name)] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
        self.df['{}_kur_q2vec'.format(model_name)] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]
        # Add word vectors as features
        for i in xrange(vector_size):
            self.df['{}_q1vec_{}'.format(model_name, i)] = question1_vectors[:, i]
            self.df['{}_q2vec_{}'.format(model_name, i)] = question2_vectors[:, i]

    def word_mover_distance(self, text1, text2):
        text1 = [w for w in str(text1).lower().split() if w not in self.stop_words]
        text2 = [w for w in str(text2).lower().split() if w not in self.stop_words]
        return self.w2c_model.wmdistance(text1, text2)

    def text2vec(self, text):
        text = word_tokenize(str(text).lower().decode('utf-8'))
        text = [w for w in text if w not in self.stop_words and w.isalpha()]
        matrix = []
        for w in text:
            try:
                matrix.append(self.w2c_model[w])
            except Exception as e:
                pass
        matrix = np.array(matrix)
        v = matrix.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())


if __name__ == '__main__':
    from csv import QUOTE_ALL
    from load_data import load_data

    df = load_data('../data/train.csv')
    fc = FeatureCreator(df)
    fc.add_word2vec_features('../models/GoogleNews-vectors-negative300.bin.gz', 'GoogleNews')
    fc.add_basic_features()
    fc.add_additional_features()
    fc.add_fuzz_features()
    print list(df)
    df.to_csv('train_features.csv', index=False, quoting=QUOTE_ALL)
