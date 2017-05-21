#!/usr/bin/env python

"""
Add more question pairs to the data
"""

import pandas as pd

from csv import QUOTE_ALL
from itertools import combinations


class QuestionPairsEnhancer(object):

    def __init__(self, df):
        self.df = df
        self.max_id = max(self.df.id) + 1
        self.filter_df()
        self.questions = self.get_questions()

    def filter_df(self):
        self.df = self.df[self.df.is_duplicate == 1]

    def get_questions(self):
        questions = {}
        for index, row in self.df.iterrows():
            questions[row['qid1']] = row['question1']
            questions[row['qid2']] = row['question2']
        return questions

    def find_naive_pairs(self):
        qid_group = {}
        group_ctr = 0
        all_pairs = set()
        print 'Grouping question pairs...'
        for index, row in self.df.iterrows():
            qid1, qid2 = row['qid1'], row['qid2']
            all_pairs.add((qid1, qid2))
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
        print 'Generating question groups...'
        groups = {}
        for qid, group in qid_group.iteritems():
            try:
                groups[group].add(qid)
            except KeyError:
                groups[group] = set([qid])
        print 'Finding new question pairs...'
        new_pairs = []
        for group, qids in groups.iteritems():
            if len(qids) > 2:
                for qid1, qid2 in combinations(qids, 2):
                    if (qid1, qid2) not in all_pairs and (qid2, qid1) not in all_pairs:
                        new_pairs.append((qid1, qid2))
        print '{} new pairs found.'.format(len(new_pairs))
        new_df = pd.DataFrame(columns=('id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'))
        for column in ['id', 'qid1', 'qid2', 'is_duplicate']:
            new_df[column] = new_df[column].astype(int)
        for i, pair in enumerate(new_pairs):
            qid1, qid2 = pair
            new_df.loc[i] = [self.max_id + i, qid1, qid2, self.questions[qid1], self.questions[qid2], 1]
        return new_df


if __name__ == '__main__':
    from load_data import load_data
    df = load_data('../data/train.csv', remove_id_columns=False)
    enhancer = QuestionPairsEnhancer(df)
    train_new = enhancer.find_naive_pairs()
    train_new.to_csv('../data/train_new.csv', index=False, quoting=QUOTE_ALL)
