# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from time import time

import sys


def progress_bar(progress, status, bar_length=20):
    """Basic progress bar, compatible with jupyter notebooks.
    """
    block = int(round(progress / 100 * bar_length))
    text = "\r[{0}] {1}% - {2}".format(
        "#" * block + "-" * (bar_length - block),
        str(int(round(progress))).rjust(3),
        status
    )
    sys.stdout.write(text)
    sys.stdout.flush()


def log_execution_time(message):
    """Print custome message and target fundtion execution time, use with

    from utils.decorators import log_execution_time

    @log_execution_time('my message')
    def foo(bar)
        do_something()

    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            ti = time()
            result = function(*args, **kwargs)
            print('[INFO] {} - {}s'.format(message, round(time() - ti, 2)))
            return result
        return wrapper
    return decorator


def compute_or_skip(feature_names):
    """Skipping computation of list of features <feature_names> unless force=True argument is there.
    Useful when rerunning (part of) the full pipeline to skip features already present, specially when they are costly.

    from utils.decorators import compute_or_skip

    @compute_or_skip(['feature1', 'feature2'])
    def compute_features_1_and_2(features)
        new_features = do_something(features)
        return new_features

    """
    def decorator(function):
        def wrapper(cls, features, *args, **kwargs):
            if not kwargs.get('force', False) and all(f in features.columns for f in feature_names):
                print(
                    '[INFO] Feature(s) "{}{}{}" found in set, skipping call.'.format(
                        feature_names[0],
                        '' if len(feature_names) == 1 else ', ' + feature_names[1],
                        '' if len(feature_names) <= 2 else ', ...',
                    )
                )
                return features
            return function(cls, features, *args, **kwargs)
        return wrapper
    return decorator
