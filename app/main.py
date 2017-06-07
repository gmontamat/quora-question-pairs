#!/usr/bin/env python

"""
Main app: each function controls a step in the process

1. clean_train(): train data enhancing, cleaning, and spell-checking
2. create_features_train(): generate features from data
3. train_model(): calibrate a model to classify train set using its features
4. clean_test(): test data cleaning and spell-checking
5. create_features_test(): generate features for test set and save in files
6. predict(): load test features and predict similarity with a trained model
"""

import pandas as pd

from csv import QUOTE_ALL, QUOTE_NONE
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# User-defined modules
from load_data import load_data
from enhancer import QuestionPairsEnhancer
from cleaner import DataCleaner
from spell_checker import QuestionSpellChecker
from features import FeatureCreator
from nnet import QuestionPairsClassifier
from xgb import XgboostClassifier

# Disable warnings when including prediction columns
pd.options.mode.chained_assignment = None

# Features used for classification
FEATURES = [
    'len_q1',
    'len_q2',
    'diff_len',
    'len_char_q1',
    'len_char_q2',
    'len_word_q1',
    'len_word_q2',
    'common_words',
    'word_match',
    'tfidf_word_match',
    'tfidf_word_match_stops',
    'jaccard_similarity',
    'word_count_diff',
    'word_count_ratio',
    'unique_word_count_diff',
    'unique_word_count_ratio',
    'unique_nonstop_word_count_diff',
    'unique_nonstop_word_count_ratio',
    'same_start',
    'char_diff',
    'char_diff_unique_nonstop',
    'total_unique_words',
    'total_unique_words_nonstop',
    'char_ratio',
    'fuzz_qratio',
    'fuzz_wratio',
    'fuzz_partial_ratio',
    'fuzz_partial_token_set_ratio',
    'fuzz_partial_token_sort_ratio',
    'fuzz_token_set_ratio',
    'fuzz_token_sort_ratio',
    'GoogleNews_norm_wmd',
    'GoogleNews_wmd',
    'GoogleNews_cosine_distance',
    'GoogleNews_cityblock_distance',
    'GoogleNews_jaccard_distance',
    'GoogleNews_canberra_distance',
    'GoogleNews_euclidean_distance',
    'GoogleNews_minkowski_distance',
    'GoogleNews_braycurtis_distance',
    'GoogleNews_skew_q1vec',
    'GoogleNews_skew_q2vec',
    'GoogleNews_kur_q1vec',
    'GoogleNews_kur_q2vec'
]
# FEATURES += ['GoogleNews_q1vec_{}'.format(i) for i in xrange(300)]
# FEATURES += ['GoogleNews_q2vec_{}'.format(i) for i in xrange(300)]


def clean_train():
    """Full cleaning and enhancing of train data
    """
    # Enhancing (add extra question pairs)
    print 'Enhancing data...'
    train = load_data('../data/train.csv', remove_id_columns=False)
    enhancer = QuestionPairsEnhancer(train)
    train_new = enhancer.find_naive_pairs()
    train_new.to_csv('../data/train_new.csv', index=False, quoting=QUOTE_ALL)
    # Cleaning
    print 'Cleaning data...'
    train = pd.concat([
        load_data('../data/train.csv', remove_id_columns=False),
        load_data('../data/train_new.csv', remove_id_columns=False)
    ], ignore_index=True)
    dc = DataCleaner(train)
    dc.clean_column('question1', 'question1_clean')
    dc.clean_column('question2', 'question2_clean')
    # Spell-checking
    print 'Spell-checking data...'
    sc = QuestionSpellChecker(train, '../dictionaries/text/')
    sc.clean_column(('question1_clean', 'question2_clean'), ('question1_sc', 'question2_sc'))
    # Save progress
    print 'Saving progress...'
    train.to_csv('../data/train_clean.csv', index=False, quoting=QUOTE_ALL)


def create_features_train():
    """Generate features to train ML model. Needs ~32Gb of RAM!
    """
    train = load_data('../data/train_clean.csv')
    fc = FeatureCreator(train, 'question1_sc', 'question2_sc')
    print 'Generating word2vec GoogleNews features...'
    fc.add_word2vec_features('../models/GoogleNews-vectors-negative300.bin.gz', 'GoogleNews')
    # print 'Generating word2vec freebase features...'
    # fc.add_word2vec_features('../models/freebase-vectors-skipgram1000-en.bin.gz', 'freebase', 1000)
    print 'Generating basic features...'
    fc.add_basic_features()
    print 'Generating additional features...'
    fc.add_additional_features()
    print 'Generating fuzzy features...'
    fc.add_fuzz_features()
    print 'Saving progress...'
    train.to_csv('../data/train_features.csv', index=False, quoting=QUOTE_ALL)


def train_model(model='nnet'):
    """Train and save model to classify question pairs
    """
    # Split train data into a 'train' set and a 'test' set used for validation
    train, test = train_test_split(pd.read_csv('../data/train_features.csv'), test_size=0.1)
    if model == 'nnet':
        qpc = QuestionPairsClassifier(
            hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', alpha=1e-6, max_iter=900
        )
        qpc.train_model(train[FEATURES].as_matrix(), train['is_duplicate'].as_matrix())
        print 'In-sample log-loss: {}'.format(qpc.neural_net.loss_)
        print 'Out-of-sample log-loss: {}'.format(log_loss(
            test['is_duplicate'].as_matrix(),
            qpc.predict_probability(test[FEATURES].as_matrix()),
            labels=[0, 1]
        ))
    elif model == 'xgboost':
        qpc = XgboostClassifier()
        qpc.train_model(
            train[FEATURES].as_matrix(), train['is_duplicate'].as_matrix(),
            test[FEATURES].as_matrix(), test['is_duplicate'].as_matrix()
        )
    else:
        raise ValueError("Model not recognized")
    print 'Saving model...'
    qpc.save_model('../models')


def clean_test():
    """Cleaning of test data
    """
    print 'Cleaning data...'
    test = load_data('../data/test.csv')
    dc = DataCleaner(test)
    dc.clean_column('question1', 'question1_clean')
    dc.clean_column('question2', 'question2_clean')
    print 'Spell-checking data...'
    sc = QuestionSpellChecker(test, '../dictionaries/text/')
    sc.clean_column(('question1_clean', 'question2_clean'), ('question1_sc', 'question2_sc'))
    print 'Saving progress...'
    test.to_csv('../data/test_clean.csv', index=False, quoting=QUOTE_ALL)


def create_features_test():
    """Load clean test set and generate its features
    """
    ctr = 0
    for test_chunk in pd.read_csv('../data/test_clean.csv', chunksize=80000):
        print 'New chunk...'
        # Generate features
        fc = FeatureCreator(test_chunk, 'question1_sc', 'question2_sc')
        print 'Generating word2vec GoogleNews features...'
        fc.add_word2vec_features('../models/GoogleNews-vectors-negative300.bin.gz', 'GoogleNews')
        print 'Generating basic features...'
        fc.add_basic_features()
        print 'Generating additional features...'
        fc.add_additional_features()
        print 'Generating fuzzy features...'
        fc.add_fuzz_features()
        # Save features for future use
        print 'Saving features...'
        test_chunk.to_csv('../data/test_features_{}.csv'.format(ctr), index=False, quoting=QUOTE_ALL)
        ctr += 1


def predict(files, model='nnet'):
    """Load test features and predict similarity of question pairs
    """
    predictions = pd.DataFrame(columns=('test_id', 'is_duplicate'))
    # Load model used for classification
    if model == 'nnet':
        qpc = QuestionPairsClassifier(model_path='../models')
    elif model == 'xgboost':
        qpc = XgboostClassifier(model_path='../models')
    else:
        raise ValueError("Model not recognized")
    # Process each feature set
    for ctr in xrange(files):
        print 'Loading features from file #{}...'.format(ctr)
        test_chunk = pd.read_csv('../data/test_features_{}.csv'.format(ctr))
        print 'Predicting...'
        test_chunk['is_duplicate'] = qpc.predict_probability(test_chunk[FEATURES].as_matrix())
        print 'Saving results...'
        predictions = pd.concat([predictions, test_chunk[['test_id', 'is_duplicate']]])
    # Save predictions
    print 'Generating submission file...'
    predictions['test_id'] = predictions['test_id'].astype(int)
    predictions.to_csv('../data/submission.csv', index=False, quoting=QUOTE_NONE)


if __name__ == '__main__':
    # clean_train()
    # create_features_train()
    train_model('xgboost')
    # clean_test()
    # create_features_test()
    # predict(30, 'xgboost')
