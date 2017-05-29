#!/usr/bin/env python

"""
Main app

1. clean_train(): data enhancing, cleaning, and spell-checking
2. create_features_train(): generate features from data
3. Train model: train ML model
3. Post-process: test data cleaning and feature creation
"""

import pandas as pd
from csv import QUOTE_ALL, QUOTE_NONE

# User-defined modules
from load_data import load_data
from enhancer import QuestionPairsEnhancer
from cleaner import DataCleaner
from spell_checker import QuestionSpellChecker
from features import FeatureCreator
from nnet import QuestionPairsClassifier


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


def train_neural_net():
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
    qpc = QuestionPairsClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
    for train_chunk in pd.read_csv('../data/train_features.csv', chunksize=50000):
        print 'New chunk...'
        x = train_chunk[features]
        y = train_chunk['is_duplicate']
        qpc.train_model(x, y)
    print 'Saving model...'
    qpc.save_model('../models')


def predict():
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
    # Load Neural Net for classification
    qpc = QuestionPairsClassifier(model_path='../models')
    # Process test set in chunks
    for test_chunk in pd.read_csv('../data/test_clean.csv', chunksize=1000):
        print 'New chunk...'
        # Generate features
        fc = FeatureCreator(test_chunk, 'question1_sc', 'question2_sc')
        print 'Generating word2vec GoogleNews features...'
        fc.add_word2vec_features('../models/GoogleNews-vectors-negative300.bin.gz', 'GoogleNews')
        print 'Generating basic features...'
        fc.add_basic_features()
        print 'Generating fuzzy features...'
        fc.add_fuzz_features()
        # Classify pair
        print 'Classifying...'
        x = test_chunk[features]
        test_chunk['is_duplicate'] = qpc.predict_probability(x)[:, 1]
        # Accumulate results
        print 'Saving results...'
        predictions = pd.concat([predictions, test_chunk[['test_id', 'is_duplicate']]])
    # Save predictions
    print 'Generating submission file...'
    predictions.to_csv('../data/submission.csv', index=False, quoting=QUOTE_NONE)


if __name__ == '__main__':
    # clean_train()
    # create_features_train()
    # clean_test()
    # train_neural_net()
    predict()
