# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sklearn import naive_bayes
from models.binary_classifier import BinaryClassifier


class GaussianNB(BinaryClassifier, naive_bayes.GaussianNB):
    pass
