# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
from datetime import timedelta

from mlencoders.target_encoder import TargetEncoder
from mlencoders.weight_of_evidence_encoder import WeightOfEvidenceEncoder

from utils.decorators import log_execution_time


# Variables to be excluded from features, typically IDs and the target itself
EXCLUDED_COLUMNS = [
    'some_id',
    'other_id',
    'target',
]
DATA_TRANSFORMS = {
    'column1':   lambda f: np.log(f.fillna(1)),
    'column2':   np.log,
}


class FeatureTransformer:

    @classmethod
    @log_execution_time('Splitting train / test sets')
    def split(cls, features, test_date, test_duration):
        columns = features.columns.difference(['target'])
        X_train = features[features['date'] < test_date].copy()
        X_test = features[
            (features['date'] >= test_date) &
            (features['date'] < test_date + timedelta(test_duration))
        ].copy()
        return X_train[columns], X_train['target'], X_test[columns], X_test['target']

    @classmethod
    @log_execution_time('Stripping columns unused in model')
    def strip(cls, features):
        return features[features.columns.difference(EXCLUDED_COLUMNS)]

    @classmethod
    def get_transform(cls, feature):
        return DATA_TRANSFORMS.get(feature, lambda f: f.fillna(0))

    @classmethod
    @log_execution_time('Transforming columns')
    def process(cls, features):
        processed = features.copy()
        for fea in processed.columns.difference(EXCLUDED_COLUMNS):
            processed[fea] = cls.get_transform(fea)(processed[fea])
        return processed

    @classmethod
    @log_execution_time('Target encoding')
    def target_encoding(cls, X, Y=None, encoder=None):
        cols = ['some_id', 'other_id']
        if encoder is None:
            encoder = TargetEncoder(cols=cols, min_samples=5, smoothing=5)
            encoder.fit(X, Y)
        encoded = encoder.transform(X).rename(columns={c: 'tgt_enc_{}'.format(c) for c in cols})
        return pd.concat([X[cols], encoded], axis=1), encoder

    @classmethod
    @log_execution_time('WOE encoding')
    def woe_encoding(cls, X, Y=None, encoder=None):
        cols = ['some_id', 'other_id']
        if encoder is None:
            encoder = WeightOfEvidenceEncoder(cols=cols, min_samples=5)
            encoder.fit(X, Y)
        encoded = encoder.transform(X).rename(columns={c: 'woe_enc_{}'.format(c) for c in cols})
        return pd.concat([X[cols], encoded], axis=1), encoder

    @classmethod
    @log_execution_time('Min-Max scaling')
    def scale(cls, features, feature_min=None, feature_max=None):
        feature_min = features.min() if feature_min is None else feature_min
        feature_max = features.max() if feature_max is None else feature_max
        # Edge case: constant features
        feature_min[feature_min == feature_max] = 0.
        feature_max[feature_min == feature_max] = 1.
        return (features - feature_min) / (feature_max - feature_min), feature_min, feature_max

    @classmethod
    def add_race_normalized_features(cls, features):
        return pd.concat(
            [
                features,
                cls.race_gap(features),
                cls.race_rank(features),
            ],
            axis=1
        )

    @classmethod
    @log_execution_time('Gap to category average')
    def race_gap(cls, features):
        cols = features.columns.difference(EXCLUDED_COLUMNS)
        avg_features = features[cols.insert(0, 'category_id')]
        avg_features = avg_features[cols] - avg_features.groupby('category_id').transform('mean')
        avg_features.columns = ['category_avg_' + c for c in avg_features.columns]
        return avg_features

    @classmethod
    @log_execution_time('Rank of features in category')
    def race_rank(cls, features):
        cols = features.columns.difference(EXCLUDED_COLUMNS)
        rank_features = features[cols.insert(0, 'category_id')]
        rank_features = rank_features.groupby('category_id').rank(pct=True)
        rank_features.columns = ['category_rank_' + c for c in rank_features.columns]
        return rank_features
