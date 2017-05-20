#!/usr/bin/env python

"""
Add more question pairs to the data
"""


class QuestionPairsEnhancer(object):

    def __init__(self, df):
        self.df = df
        self.filter_df()

    def filter_df(self):
        self.df = self.df[self.df.is_duplicate == 1]

    def find_naive_pairs(self):
        qid_group = {}
        group_ctr = 0
        for index, row in self.df.iterrows():
            qid1, qid2 = row['qid1'], row['qid2']
            if qid1 in qid_group:
                if qid2 in qid_group:
                    pass
                else:
                    qid_group[qid2] = qid_group[qid1]
            elif qid2 in qid_group:
                qid_group[qid1] = qid_group[qid2]
            else:
                group_ctr += 1
                qid_group[qid1] = group_ctr
                qid_group[qid2] = group_ctr
        print group_ctr


if __name__ == '__main__':
    from load_data import load_data
    df = load_data('../data/train.csv', remove_id_columns=False)
    enhancer = QuestionPairsEnhancer(df)
    enhancer.find_naive_pairs()
    import pdb
    pdb.set_trace()
