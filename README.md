# Quora Question Pairs

Analysis and submissions code for the Kaggle competition. The idea is to identify question pairs that have the same intent.

## Ideas

### Data Cleaning

* Text cleaning
    - https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text/notebook
    - https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text/comments
* Remove common words, stop words, and punctuation
* Stem words (keep only the root)
* Spell Checker
    - https://github.com/mattalcock/blog/blob/master/2012/12/5/python-spell-checker.rst
    - http://stackoverflow.com/questions/40188226/nltks-spell-checker-is-not-working-correctly
* Word tagging (verb, noun, etc.)

### Prediction

* https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur
* https://www.kaggle.com/c/quora-question-pairs/discussion/30260
* https://www.kaggle.com/c/quora-question-pairs/discussion/30340
* https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
* TF-IDF
    - http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/
    - http://blog.christianperone.com/2011/09/machine-learning-text-feature-extraction-tf-idf-part-i/
    - http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/

## Useful links

* https://www.kaggle.com/c/quora-question-pairs
* EDA (Exploratory Data Analysis)
    - https://www.kaggle.com/sudalairajkumar/quora-question-pairs/simple-exploration-notebook-quora-ques-pair/notebook

## Timeline

* May 30, 2017 - Pretrained model posting deadline.
* May 30, 2017 - Entry deadline. You must accept the competition rules before this date in order to compete.
* May 30, 2017 - Team Merger deadline. This is the last day participants may join or merge teams.
* June 6, 2017 - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.
