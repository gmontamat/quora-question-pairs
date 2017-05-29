#!/usr/bin/env python

"""
Train a Neural Network model on train set
"""

import os
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, Imputer


class QuestionPairsClassifier(object):

    def __init__(self, model_path=None, hidden_layer_sizes=(100, 100),
                 activation='tanh', solver='adam', alpha=1e-7, max_iter=1000):
        if model_path:
            self.neural_net, self.imputer, self.scaler = self.load_model(model_path)
            self.ready = True
        else:
            self.neural_net = MLPClassifier(hidden_layer_sizes, activation, solver, alpha, max_iter)
            self.imputer = Imputer()
            self.scaler = StandardScaler()
            self.ready = False

    def train_model(self, x, y):
        print 'Fixing NaNs in train data...'
        x = self.imputer.fit_transform(x)
        print 'Scaling train data...'
        self.scaler.partial_fit(x)
        x = self.scaler.transform(x)
        print 'Fitting Neural Net...'
        if not self.ready:
            self.neural_net.partial_fit(x, y, classes=np.array([0, 1]))
            self.ready = True
        else:
            self.neural_net.partial_fit(x, y)

    @staticmethod
    def load_model(model_path):
        neural_net = joblib.load(os.path.join(model_path, 'neural_net.pkl'))
        imputer = joblib.load(os.path.join(model_path, 'imputer.pkl'))
        scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        return neural_net, imputer, scaler

    def save_model(self, model_path):
        if self.ready:
            joblib.dump(self.neural_net, os.path.join(model_path, 'neural_net.pkl'))
            joblib.dump(self.imputer, os.path.join(model_path, 'imputer.pkl'))
            joblib.dump(self.scaler, os.path.join(model_path, 'scaler.pkl'))
        else:
            raise ValueError("Model not fitted")

    def predict_probability(self, x):
        if self.ready:
            x = self.imputer.fit_transform(x)
            x = self.scaler.transform(x)
            return self.neural_net.predict_proba(x)
        raise ValueError("Model not fitted")


if __name__ == '__main__':
    features = [
        'len_q1', 'len_q2', 'diff_len', 'len_char_q1', 'len_char_q2', 'len_word_q1', 'len_word_q2', 'common_words',
        'fuzz_qratio', 'fuzz_wratio', 'fuzz_partial_ratio', 'fuzz_partial_token_set_ratio',
        'fuzz_partial_token_sort_ratio', 'fuzz_token_set_ratio', 'fuzz_token_sort_ratio', 'GoogleNews_norm_wmd',
        'GoogleNews_wmd', 'GoogleNews_cosine_distance', 'GoogleNews_cityblock_distance', 'GoogleNews_jaccard_distance',
        'GoogleNews_canberra_distance', 'GoogleNews_euclidean_distance', 'GoogleNews_minkowski_distance',
        'GoogleNews_braycurtis_distance', 'GoogleNews_skew_q1vec', 'GoogleNews_skew_q2vec', 'GoogleNews_kur_q1vec',
        'GoogleNews_kur_q2vec'
    ]
    features += ['GoogleNews_q1vec_{}'.format(i) for i in xrange(300)]
    features += ['GoogleNews_q2vec_{}'.format(i) for i in xrange(300)]
    qpc = QuestionPairsClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
    for train_chunk in pd.read_csv('../data/train_features.csv', chunksize=50000):
        print 'New chunk...'
        x = train_chunk[features]
        y = train_chunk['is_duplicate']
        qpc.train_model(x, y)
    print 'Saving model...'
    qpc.save_model('../models')
