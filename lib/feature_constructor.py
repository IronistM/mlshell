# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd

from utils.decorators import compute_or_skip
from utils.decorators import log_execution_time


# These bounds are based on (visual) data exploration
DATA_BOUNDS = {
    'column1': (1, 5 * 1e5),
    'column2': (0, 1e6),
}


class FeatureConstructor:

    @classmethod
    @log_execution_time('Building core features')
    def core_features(cls, dataset):
        features = dataset['table1']\
            .merge(dataset['table2'], how='left', on='some_id', suffixes=['_table1', '_table2'])
        return features

    @classmethod
    @log_execution_time('Filtering data')
    def filter(cls, features):
        mask = features['col'] == 1
        for (column, (l, h)) in DATA_BOUNDS.items():
            values = features[column].fillna(0)
            mask &= (values >= l) & (values < h)

        filtered = features[mask]
        print(
            '[INFO] Data size {} > {} (-{}%)'.format(
                features.shape[0],
                filtered.shape[0],
                round(100 - filtered.shape[0] / features.shape[0] * 100), 1,
            )
        )
        return filtered

    @classmethod
    def add_all_features(cls, features, dataset):
        features = cls.add_target(features)                                     # Label - value to predict
        features = cls.add_horse_race_counts(features)                          # Race counts for horses
        features = cls.add_jockey_race_counts(features)                         # Race counts for jockeys
        features = cls.add_owner_race_counts(features)                          # Race counts for owners
        features = cls.add_coach_race_counts(features)                          # Race counts for coaches
        features = cls.add_horse_last_race(features, 'horse')                   # Last race result for horses
        features = cls.add_jockey_last_race(features, 'jockey')                 # Last race result for jockeys
        features = cls.add_owner_last_race(features, 'owner')                   # Last race result for owners
        features = cls.add_coach_last_race(features, 'coach')                   # Last race result for coaches
        features = cls.add_horse_last_n_races(features, 'horse')                # Last n races results for horses
        features = cls.add_jockey_last_n_races(features, 'jockey')              # Last n races results for jockeys
        features = cls.add_owner_last_n_races(features, 'owner')                # Last n races results for owners
        features = cls.add_coach_last_n_races(features, 'coach')                # Last n races results for coaches
        return features

    @classmethod
    @compute_or_skip(['target'])
    def add_target(cls, features, force=False):
        features = features.assign(target=features['rank'] <= 3)
        return features

    @classmethod
    @compute_or_skip(['horse_n_wins', 'horse_n_races'])
    @log_execution_time('Adding race counts for horses')
    def add_horse_race_counts(cls, features, force=False):
        return cls.add_race_counts(features, 'horse')

    @classmethod
    @compute_or_skip(['jockey_n_wins', 'jockey_n_races'])
    @log_execution_time('Adding race counts for jockeys')
    def add_jockey_race_counts(cls, features, force=False):
        return cls.add_race_counts(features, 'jockey')

    @classmethod
    @compute_or_skip(['owner_n_wins', 'owner_n_races'])
    @log_execution_time('Adding race counts for owners')
    def add_owner_race_counts(cls, features, force=False):
        return cls.add_race_counts(features, 'owner')

    @classmethod
    @compute_or_skip(['coach_n_wins', 'coach_n_races'])
    @log_execution_time('Adding race counts for coaches')
    def add_coach_race_counts(cls, features, force=False):
        return cls.add_race_counts(features, 'coach')

    @classmethod
    def add_race_counts(cls, features, instance_name):
        counts = features[['date', '{}_id'.format(instance_name), 'target']].copy()
        counts['{}_n_wins'.format(instance_name)] = counts['target']
        counts['{}_n_races'.format(instance_name)] = 1
        counts.drop('target', axis=1, inplace=True)
        counts = counts.groupby(by=['{}_id'.format(instance_name), 'date']).sum()
        counts = counts.groupby(level=[0]).cumsum()
        counts = np.sqrt(counts.groupby(level=[0]).shift(periods=1, axis=0).fillna(0))
        counts = counts.reset_index()
        features = features.merge(counts, how='left', on=['{}_id'.format(instance_name), 'date'])
        return features

    @classmethod
    @compute_or_skip(['horse_last_race'])
    @log_execution_time('Adding last race result for horses')
    def add_horse_last_race(cls, features, force=False):
        return cls.last_race_result(features, 'horse')

    @classmethod
    @compute_or_skip(['jockey_last_race'])
    @log_execution_time('Adding last race result for jockeys')
    def add_jockey_last_race(cls, features, force=False):
        return cls.last_race_result(features, 'jockey')

    @classmethod
    @compute_or_skip(['owner_last_race'])
    @log_execution_time('Adding last race result for owners')
    def add_owner_last_race(cls, features, force=False):
        return cls.last_race_result(features, 'owner')

    @classmethod
    @compute_or_skip(['coach_last_race'])
    @log_execution_time('Adding last race result for coaches')
    def add_coach_last_race(cls, features, force=False):
        return cls.last_race_result(features, 'coach')

    @classmethod
    def last_race_result(cls, features, instance_name):
        counts = features[['date', '{}_id'.format(instance_name), 'target']].copy()
        counts['{}_last_race'.format(instance_name)] = counts['target']
        counts.drop('target', axis=1, inplace=True)
        counts = counts.groupby(by=['{}_id'.format(instance_name), 'date']).sum()
        counts = counts.groupby(level=[0]).shift(periods=1, axis=0).fillna(0)
        counts = counts.reset_index()
        features = features.merge(counts, how='left', on=['{}_id'.format(instance_name), 'date'])
        return features

    @classmethod
    @compute_or_skip(['horse_last_n_races'])
    @log_execution_time('Adding last 3 races result for horses')
    def add_horse_last_n_races(cls, features, force=False):
        return cls.last_n_wins(features, 'horse', 3)

    @classmethod
    @compute_or_skip(['jockey_last_n_races'])
    @log_execution_time('Adding last 3 races result for jockeys')
    def add_jockey_last_n_races(cls, features, force=False):
        return cls.last_n_wins(features, 'jockey', 3)

    @classmethod
    @compute_or_skip(['owner_last_n_races'])
    @log_execution_time('Adding last 3 races result for owners')
    def add_owner_last_n_races(cls, features, force=False):
        return cls.last_n_wins(features, 'owner', 3)

    @classmethod
    @compute_or_skip(['coach_last_n_races'])
    @log_execution_time('Adding last 3 races result for coaches')
    def add_coach_last_n_races(cls, features, force=False):
        return cls.last_n_wins(features, 'coach', 3)

    @classmethod
    def last_n_wins(cls, features, instance_name, window):
        counts = features[['date', '{}_id'.format(instance_name), 'target']].copy()
        counts['{}_last_n_races'.format(instance_name)] = counts['target']
        counts.drop('target', axis=1, inplace=True)
        counts = counts.groupby(by=['{}_id'.format(instance_name), 'date']).sum()
        counts = counts.groupby(level=[0]).apply(lambda x: x.rolling(window=window, min_periods=1).sum())
        counts = counts.groupby(level=[0]).shift(periods=1, axis=0).fillna(0)
        counts = counts.reset_index()
        features = features.merge(counts, how='left', on=['{}_id'.format(instance_name), 'date'])
        return features
