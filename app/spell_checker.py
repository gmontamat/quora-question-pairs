#!/usr/bin/env python

"""
Spell checks text
"""

import collections
import re
import string


class SpellChecker(object):
    """Simple spell checker based on Peter Norvig's
    implementation available on http://norvig.com/spell-correct.html
    """

    def __init__(self, words_file):
        words = self.get_words(open(words_file).read())
        self.dictionary = collections.Counter(words)
        self.n_words = float(sum(self.dictionary.values()))

    @staticmethod
    def get_words(text):
        """Find words in text file
        """
        return re.findall(r'\w+', text.lower())

    def double_edit(self, word):
        """Get set of words that are two edits away from 'word'
        """
        return set(e2 for e1 in self.single_edit(word) for e2 in self.single_edit(e1))

    @staticmethod
    def single_edit(word):
        """Get set of words that are one edit away from 'word'
        """
        letters = string.ascii_lowercase
        splits = [(word[:i], word[i:]) for i in xrange(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def word_probability(self, word):
        """Compute estimated probability of a word, P(w)
        """
        return self.dictionary[word] / self.n_words

    def known_words(self, words):
        """Return subset of 'words' that appear in the dictionary
        """
        return set(word for word in words if word in self.dictionary)

    def get_candidates(self, word):
        """Generate possible spelling correction for word
        """
        candidates = (
            self.known_words([word]) or
            self.known_words(self.single_edit(word)) or
            self.known_words(self.double_edit(word)) or
            set([word])
        )
        return candidates

    def correct(self, word):
        """Correct a word using most probable spelling correction
        """
        return max(self.get_candidates(word), key=self.word_probability)


if __name__ == '__main__':
    sc = SpellChecker('../dictionaries/big.txt')
    print sc.get_candidates('whatsover')
    print sc.correct('whatsover')
