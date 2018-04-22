# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import warnings

import MySQLdb
import pandas as pd
import yaml


CREDENTIALS = '/../config/database.yml'
DEFAULT_ENV = 'development'


class MySQL:

    @classmethod
    def query(cls, query, env=DEFAULT_ENV):
        """
        Executes a query on our MySQL databases.
        :param string query: query to execute
        :param string env: environment for which we want to query the databases.
        :return: table containing the result of the query
        :rtype: pd.DataFrame
        """
        # loading database credentials
        with open(os.path.dirname(os.path.realpath(__file__)) + CREDENTIALS, 'r') as stream:
            credentials = yaml.load(stream)
        # verifying if environment passed in argument exists
        if env not in credentials:
            warnings.warn("Invalid environment {}. Defaulting to {}.".format(env, DEFAULT_ENV))
            env = DEFAULT_ENV
        # querying data
        db = MySQLdb.connect(
            host=credentials[env]['host'],
            user=credentials[env]['username'],
            passwd=credentials[env]['password'],
            db=credentials[env]['database'],
            charset=credentials[env]['encoding'],
            use_unicode=True
        )
        df = pd.read_sql(query, db)
        db.close()
        return df
