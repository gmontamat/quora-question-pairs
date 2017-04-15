#!/usr/bin/env python

"""
Creates features from data
"""

import fuzzywuzzy.fuzz as fuzz


class FeatureCreator(object):

    def __init__(self, df):
        self.df = df

    def add_basic_features(self):
        self.df['len_q1'] = self.df.question1.apply(lambda x: len(str(x)))
        self.df['len_q2'] = self.df.question2.apply(lambda x: len(str(x)))
        self.df['diff_len'] = self.df.len_q1 - self.df.len_q1
        self.df['len_char_q1'] = self.df.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        self.df['len_char_q2'] = self.df.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        self.df['len_word_q1'] = self.df.question1.apply(lambda x: len(str(x).split()))
        self.df['len_word_q2'] = self.df.question2.apply(lambda x: len(str(x).split()))
        self.df['common_words'] = self.df.apply(
            lambda x: len(
                set(str(x['question1']).lower().split()).intersection(
                    set(str(x['question2']).lower().split())
                )
            ), axis=1
        )

    def add_fuzz_features(self):
        self.df['fuzz_qratio'] = self.df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
        self.df['fuzz_wratio'] = self.df.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
        self.df['fuzz_partial_ratio'] = self.df.apply(
            lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1
        )
        self.df['fuzz_partial_token_set_ratio'] = self.df.apply(
            lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1
        )
        self.df['fuzz_partial_token_sort_ratio'] = self.df.apply(
            lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1
        )
        self.df['fuzz_token_set_ratio'] = self.df.apply(
            lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1
        )
        self.df['fuzz_token_sort_ratio'] = self.df.apply(
            lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1
        )


if __name__ == '__main__':
    from load_data import load_data
    df = load_data('../data/train.csv')
    fc = FeatureCreator(df)
    fc.add_basic_features()
    fc.add_fuzz_features()
    print list(df)
