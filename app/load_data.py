#!/usr/bin/env python

"""
Load CSV files
"""

import os
import pandas as pd


def read_csv(file_path):
    return pd.read_csv(file_path, sep=',')


if __name__ == '__main__':
    print read_csv('../data/test.csv')
