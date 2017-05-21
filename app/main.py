#!/usr/bin/env python

"""
Main app

1. Pre-process: train data enhancing, cleaning, spell-checking, and feature creation
2. Process: train ML model
3. Post-process: test data cleaning and feature creation
"""

import pandas as pd
from csv import QUOTE_ALL

# User-defined modules
from load_data import load_data
from enhancer import QuestionPairsEnhancer
from cleaner import DataCleaner
from spell_checker import QuestionSpellChecker


def pre_process():
    """Generate features 
    """
    # Enhancing (add extra question pairs)
    print 'Enhancing data...'
    train = load_data('../data/train.csv', remove_id_columns=False)
    enhancer = QuestionPairsEnhancer(train)
    train_new = enhancer.find_naive_pairs()
    train_new.to_csv('../data/train_new.csv', index=False, quoting=QUOTE_ALL)
    # Cleaning
    print 'Cleaning data...'
    train = pd.concat([load_data('../data/train.csv'), load_data('../data/train_new.csv')])
    dc = DataCleaner(train)
    dc.clean_column('question1', 'question1_clean')
    dc.clean_column('question2', 'question2_clean')
    print 'Saving progress...'
    train.to_csv('../data/train_clean1.csv', index=False, quoting=QUOTE_ALL)
    # Spell-checking
    print 'Spell-checking data...'
    sc = QuestionSpellChecker(train, '../dictionaries/text/')
    sc.clean_column(('question1_clean', 'question2_clean'), ('question1_sc', 'question2_sc'))
    # Save progress
    print 'Saving progress...'
    train.to_csv('../data/train_clean.csv', index=False, quoting=QUOTE_ALL)


if __name__ == '__main__':
    pre_process()
