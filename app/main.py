#!/usr/bin/env python

"""
Main app: each function controls a step in the process

1. clean_train(): train data enhancing, cleaning, and spell-checking
2. create_features_train(): generate features from data
3. clean_test(): test data cleaning and spell-checking
4. train_model(): calibrate a model to classify train set using its features
5. predict(): generate features on test set and use trained model to predict similarity
6. predict_again(): same as above but uses pre-computed features
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
    ])
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
    """Generate features to train ML model. Needs +16Gb of RAM!
    """
    train = load_data('../data/train_clean.csv')
    fc = FeatureCreator(train, 'question1_sc', 'question2_sc')
    print 'Generating word2vec GoogleNews features...'
    fc.add_word2vec_features('../models/GoogleNews-vectors-negative300.bin.gz', 'GoogleNews')
    # print 'Generating word2vec freebase features...'
    # fc.add_word2vec_features('../models/freebase-vectors-skipgram1000-en.bin.gz', 'freebase', 1000)
    print 'Generating basic features...'
    fc.add_basic_features()
    print 'Generating fuzzy features...'
    fc.add_fuzz_features()
    print 'Saving progress...'
    train.to_csv('../data/train_features.csv', index=False, quoting=QUOTE_ALL)


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


def train_model(model='nnet'):
    """Train and save model to classify question pairs
    """
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
    # features += ['q1_wv2_{}'.format(i + 1) for i in xrange(300)]
    # features += ['q2_wv2_{}'.format(i + 1) for i in xrange(300)]
    # Load train and test sets
    train, test = train_test_split(pd.read_csv('../data/train_features.csv', nrows=1000), test_size=0.1)
    # train = pd.read_csv('../data/train_features.csv.gz', compression='gzip', nrows=367549)
    # test =  pd.read_csv('../data/train_features.csv.gz', compression='gzip', nrows=36754, skiprows=range(1,367550))
    if model == 'nnet':
        qpc = QuestionPairsClassifier(
            hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', alpha=1e-6, max_iter=900
        )
        qpc.train_model(train[features].as_matrix(), train['is_duplicate'].as_matrix())
        print 'In-sample log-loss: {}'.format(qpc.neural_net.loss_)
        print 'Out-of-sample log-loss: {}'.format(log_loss(
            test['is_duplicate'].as_matrix(),
            qpc.predict_probability(test[features].as_matrix()),
            labels=[0, 1]
        ))
    elif model == 'xgboost':
        qpc = XgboostClassifier()
        qpc.train_model(
            train[features].as_matrix(), train['is_duplicate'].as_matrix(),
            test[features].as_matrix(), test['is_duplicate'].as_matrix()
        )
    else:
        raise ValueError("Model not recognized")
    print 'Saving model...'
    qpc.save_model('../models')


def predict(model='nnet'):
    """Load clean test set, generate its features, and predict similarity
    using a trained neural net"""
    predictions = pd.DataFrame(columns=('test_id', 'is_duplicate'))
    # Features used for classification
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
    # Load model used for classification
    if model == 'nnet':
        qpc = QuestionPairsClassifier(model_path='../models')
    elif model == 'xgboost':
        qpc = XgboostClassifier(model_path='../models')
    else:
        raise ValueError("Model not recognized")
    # Process test set in chunks
    ctr = 0
    for test_chunk in pd.read_csv('../data/test_clean.csv', chunksize=80000):
        print 'New chunk...'
        # Generate features
        fc = FeatureCreator(test_chunk, 'question1_sc', 'question2_sc')
        print 'Generating word2vec GoogleNews features...'
        fc.add_word2vec_features('../models/GoogleNews-vectors-negative300.bin.gz', 'GoogleNews')
        print 'Generating basic features...'
        fc.add_basic_features()
        print 'Generating fuzzy features...'
        fc.add_fuzz_features()
        # Save features for future use
        print 'Saving features...'
        test_chunk.to_csv('../data/test_features_{}.csv'.format(ctr), index=False, quoting=QUOTE_ALL)
        # Classify pair
        print 'Classifying...'
        test_chunk['is_duplicate'] = qpc.predict_probability(test_chunk[features].as_matrix())
        # Accumulate results
        print 'Saving results...'
        predictions = pd.concat([predictions, test_chunk[['test_id', 'is_duplicate']]])
        ctr += 1
    # Save predictions
    print 'Generating submission file...'
    predictions['test_id'] = predictions['test_id'].astype(int)
    predictions.to_csv('../data/submission.csv', index=False, quoting=QUOTE_NONE)


def predict_again(files, model='nnet'):
    """Load features and predict similarity
    """
    predictions = pd.DataFrame(columns=('test_id', 'is_duplicate'))
    # Features used for classification
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
    # Load model for classification
    if model == 'nnet':
        qpc = QuestionPairsClassifier(model_path='../models')
    elif model == 'xgboost':
        qpc = XgboostClassifier(model_path='../models')
    else:
        raise ValueError("Model not recognized")
    # Process each feature set
    for ctr in xrange(files):
        print 'Loading features from file #{}...'.format(ctr)
        test_chunk = pd.read_csv('../data/test_features_{}.csv.gz'.format(ctr), compression='gzip')
        print 'Predicting...'
        test_chunk['is_duplicate'] = qpc.predict_probability(test_chunk[features].as_matrix())
        print 'Saving results...'
        predictions = pd.concat([predictions, test_chunk[['test_id', 'is_duplicate']]])
    # Save predictions
    print 'Generating submission file...'
    predictions['test_id'] = predictions['test_id'].astype(int)
    predictions.to_csv('../data/submission.csv', index=False, quoting=QUOTE_NONE)


if __name__ == '__main__':
    # clean_train()
    # create_features_train()
    # clean_test()
    train_model('xgboost')
    # predict()
    # predict_again(30, 'xgboost')
