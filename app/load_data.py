#!/usr/bin/env python

"""
Load data files
"""

import pandas as pd


def load_data(file_path, remove_id_columns=True):
    df = pd.read_csv(file_path)
    if remove_id_columns:
        remove_columns = [column for column in ['id', 'qid1', 'qid2'] if column in df]
        df = df.drop(remove_columns, axis=1)
    return df


if __name__ == '__main__':
    train = load_data('../data/train.csv')
    print list(train)
