#!/usr/bin/env python

"""
Main app

1. clean_train(): data enhancing, cleaning, and spell-checking
2. create_features_train(): generate features from data
3. Train model: train ML model
3. Post-process: test data cleaning and feature creation
"""

import pandas as pd
from csv import QUOTE_ALL

# User-defined modules
from load_data import load_data
from enhancer import QuestionPairsEnhancer
from cleaner import DataCleaner
from spell_checker import QuestionSpellChecker
from features import FeatureCreator


def clean_train():
    """Full cleaning and enhancing of data 
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
    """Generate features to train ML model
    Needs +16Gb of RAM!
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


if __name__ == '__main__':
    # clean_train()
    # create_features_train()
    clean_test()
