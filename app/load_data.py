#!/usr/bin/env python

"""
Load data files
"""

import pandas as pd


def load_data(file_path):
    df = pd.read_csv(file_path)
    try:
        # Unnecessary fields in training data
        df = df.drop(['id', 'qid1', 'qid2'], axis=1)
    except ValueError:
        pass
    return df


if __name__ == '__main__':
    train = load_data('../data/train.csv')
    print list(train)
