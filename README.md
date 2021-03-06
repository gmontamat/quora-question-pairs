# Quora Question Pairs

Analysis and submissions code for the Kaggle competition. The idea is to identify question pairs that have the same intent.
This code doesn't use the "leaky" features mentioned on Kaggle!

## Ideas

### Data Cleaning

* Text cleaning
    - https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text/notebook
    - https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text/comments
* Remove common words, stop words, and punctuation
* Stem words (keep only the root)
* Spell Checker
    - http://norvig.com/spell-correct.html
    - https://github.com/mattalcock/blog/blob/master/2012/12/5/python-spell-checker.rst
    - http://stackoverflow.com/questions/40188226/nltks-spell-checker-is-not-working-correctly
    - Excellent dictionary source: https://dumps.wikimedia.org/enwiki/20170420/
    - To extract text from Wikipedia dumps: https://github.com/attardi/wikiextractor
* Word tagging (verb, noun, etc.)

### Prediction and features

* Several approaches
    - https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur
    - http://www.erogol.com/duplicate-question-detection-deep-learning/
    - https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
    - https://www.kaggle.com/c/quora-question-pairs/discussion/30260
    - https://www.kaggle.com/c/quora-question-pairs/discussion/30340
    - https://www.kaggle.com/act444/lb-0-158-xgb-handcrafted-leaky
* Word2Vec
    - https://code.google.com/archive/p/word2vec/
	- http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
	- http://radimrehurek.com/gensim/index.html
	- https://rare-technologies.com/word2vec-tutorial/
* TF-IDF
    - http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/
    - http://blog.christianperone.com/2011/09/machine-learning-text-feature-extraction-tf-idf-part-i/
    - http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/

### Models

* Neural networks
    - http://scikit-learn.org/stable/modules/neural_networks_supervised.html
* Xgboost
    - http://xgboost.readthedocs.io/en/latest/
    - Kernel example: https://www.kaggle.com/hbaflast/lb-0-158-xgb-handcrafted-leaky/code

## Useful links

* https://www.kaggle.com/c/quora-question-pairs
* EDA (Exploratory Data Analysis)
    - https://www.kaggle.com/sudalairajkumar/quora-question-pairs/simple-exploration-notebook-quora-ques-pair/notebook
* https://www.kaggle.com/shubh24/d/quora/question-pairs-dataset/everything-you-wanna-know/notebook

## Timeline

* May 30, 2017 - Pretrained model posting deadline.
* May 30, 2017 - Entry deadline. You must accept the competition rules before this date in order to compete.
* May 30, 2017 - Team Merger deadline. This is the last day participants may join or merge teams.
* June 6, 2017 - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.
