# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sklearn import ensemble
from models.binary_classifier import BinaryClassifier


class RandomForestClassifier(BinaryClassifier, ensemble.RandomForestClassifier):
    pass
