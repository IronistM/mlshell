# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sklearn import metrics

import numpy as np
import pandas as pd

import json


class BinaryClassifier:

    def __init__(self, **kwargs):
        self.params = kwargs
        self.scores_train = {}
        self.scores_test = {}
        self.sklearn_class = self.__class__.__bases__[-1]
        self.sklearn_class.__init__(self, n_jobs=-1, **kwargs)

    def fit(self, x, y):
        self.sklearn_class.fit(self, x, y)
        self.scores_train = self.score(y, self.predict_proba(x), train=True)

    def predict_proba(self, x):
        return self.sklearn_class.predict_proba(self, x)[:, 1]

    def score(self, y, probas, train=False):
        scores = {
            method.__name__: method(y, probas)
            for method in [self.roc_auc_score, self.f1_score, self.accuracy_score, self.log_loss]
        }
        if not train:
            self.scores_test = scores
        return scores

    def roc_auc_score(self, y, probas):
        return metrics.roc_auc_score(y, probas)

    def f1_score(self, y, probas, thresh=0.5):
        return metrics.f1_score(y, probas >= thresh)

    def accuracy_score(self, y, probas, thresh=0.5):
        return metrics.accuracy_score(y, probas >= thresh)

    def log_loss(self, y, probas):
        return metrics.log_loss(y, probas)

    def hash(self):
        return hash(json.dumps(self.params, sort_keys=True))

    def save_results(self):
        """Useful for a grid search for example.
        """
        if not self.scores_train:
            print('Model must be fitted first, skipping.')
            return
        if not self.scores_test:
            print('Model must be scored on test set first, skipping.')
            return
        with open(self.__class__.__name__ + '.txt', 'a') as outfile:
            json.dump(
                (
                    self.hash(),
                    {
                        'params': self.params,
                        'scores_train': self.scores_train,
                        'scores_test': self.scores_test,
                    },
                ),
                outfile,
            )
            outfile.write('\n')

    def compute_reward(self, some_arguments):
        """To be defined (and probably used!) when the outcome of a prediction is associated with a reward.
        """
        pass
