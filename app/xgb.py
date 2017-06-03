#!/usr/bin/env python

"""
Train a Xgboost model on train set
"""

import os

from sklearn.externals import joblib
from sklearn.preprocessing import Imputer
from xgboost import XGBClassifier


class XgboostClassifier(object):

    def __init__(self, model_path=None):
        if model_path:
            self.model = self.load_model(model_path)
            self.ready = True
        else:
            self.model = XGBClassifier()
            self.imputer = Imputer()
            self.ready = False

    def train_model(self, x, y):
        # Fix NaNs in train data
        x = self.imputer.fit_transform(x)
        # Fit model
        self.model.fit(x, y)
        self.ready = True

    @staticmethod
    def load_model(model_path):
        return joblib.load(os.path.join(model_path, 'xgboost.pkl'))

    def save_model(self, model_path):
        if not self.ready:
            raise ValueError("Model not fitted")
        joblib.dump(self.model, os.path.join(model_path, 'xgboost.pkl'))

    def predict_probability(self, x):
        if not self.ready:
            raise AttributeError("Model not fitted")
        x = self.imputer.fit_transform(x)
        return self.model.predict_proba(x)

    def predict(self, x):
        if not self.ready:
            raise AttributeError("Model not fitted")
        x = self.imputer.fit_transform(x)
        return self.model.predict(x)
