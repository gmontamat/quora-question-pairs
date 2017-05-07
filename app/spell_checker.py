#!/usr/bin/env python

"""
Spell checks text
"""

import os
import math
import re
import string

from collections import Counter


class SpellChecker(object):
    """Simple spell checker based on Peter Norvig's
    implementation available on http://norvig.com/spell-correct.html
    """

    def __init__(self, words_file):
        words = self.get_words(open(words_file).read())
        self.dictionary = Counter(words)
        self.logn_words = math.log(float(sum(self.dictionary.values())))

    @staticmethod
    def get_words(text):
        """Get words from 'text'
        """
        return re.findall(r'\w+', text.lower())

    def double_edit(self, word):
        """Get set of words that are two edits away from 'word'
        """
        return (e2 for e1 in self.single_edit(word) for e2 in self.single_edit(e1))

    @staticmethod
    def single_edit(word):
        """Get set of words that are one edit away from 'word'
        """
        letters = string.ascii_lowercase
        splits = [(word[:i], word[i:]) for i in xrange(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters + ' ']
        return set(deletes + transposes + replaces + inserts)

    def log_probability(self, word):
        try:
            return math.log(self.dictionary[word])
        except ValueError:
            return -750.0

    def word_probability(self, word):
        """Compute estimated probability of a word, P(w)
        """
        return sum([self.log_probability(w) - self.logn_words for w in word.split()])

    def known_words(self, words):
        """Return subset of 'words' that appear in the dictionary
        """
        return set(word for word in words if all([w in self.dictionary for w in word.split()]))

    def get_candidates(self, word):
        """Generate possible spelling correction for word
        """
        candidates = (
            self.known_words([word]) or
            self.known_words(self.single_edit(word)) or
            self.known_words(self.double_edit(word)) or
            [word]
        )
        return candidates

    def correct(self, word):
        """Correct a word using most probable spelling correction
        """
        return max(self.get_candidates(word), key=self.word_probability)

    def clean_questions(self, question1, question2):
        """Correct misspellings in a question using a possible matching one
        """
        words1 = self.get_words(question1)
        words2 = self.get_words(question2)
        checked_words1 = []
        checked_words2 = []
        for word in words1:
            if word in self.dictionary or word in words2:
                checked_words1.append(word)
                continue
            for candidate in self.get_candidates(word):
                if candidate in words2:
                    checked_words1.append(candidate)
                    print '{} ==> {}'.format(word, candidate)
                    break
            else:
                correction = self.correct(word)
                print '{} ==> {}'.format(word, correction)
                checked_words1.append(correction)
        for word in words2:
            if word in self.dictionary or word in words1:
                checked_words2.append(word)
                continue
            for candidate in self.get_candidates(word):
                if candidate in words1:
                    checked_words2.append(candidate)
                    print '{} ==> {}'.format(word, candidate)
                    break
            else:
                correction = self.correct(word)
                print '{} ==> {}'.format(word, correction)
                checked_words2.append(correction)
        return ' '.join(checked_words1), ' '.join(checked_words2)


class WikiSpellChecker(SpellChecker):
    """Spell checker that obtains vocabulary from a Wikipedia dump
    """

    def __init__(self, wiki_path):
        self.dictionary = self.load_wiki_dump(wiki_path)
        self.logn_words = math.log(float(sum(self.dictionary.values())))

    def load_wiki_dump(self, wiki_path):
        dictionary = Counter()
        wiki_files = [
            os.path.join(path, file_name)
            for path, subdir, files in os.walk(wiki_path)
            for file_name in files if 'wiki_' in file_name
        ]
        for wiki_file in wiki_files:
            text = open(wiki_file).read()
            pattern = re.compile(r'<.*?>')
            text = pattern.sub('', text)
            dictionary.update(self.get_words(text))
        # Filter dictionary since misspellings are possible
        dictionary = Counter({word: count for word, count in dictionary.iteritems() if count > 3})
        return dictionary


if __name__ == '__main__':
    # sc = SpellChecker('../dictionaries/big.txt')
    sc = WikiSpellChecker('../dictionaries/text/')
    print sc.correct('whatsover')
    print sc.correct('verfify')
    print sc.correct('nintendo')

    from load_data import load_data
    from cleaner import DataCleaner

    train = load_data('../data/train.csv')
    dc = DataCleaner(train)
    dc.clean_column('question1')
    dc.clean_column('question2')
    for question1, question2 in zip(train.question1, train.question2):
        sc.clean_questions(question1, question2)
