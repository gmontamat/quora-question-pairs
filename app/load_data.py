#!/usr/bin/env python

"""
Load data files
"""

import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


if __name__ == '__main__':
    train = load_data('../data/train.csv')
    print list(train)
