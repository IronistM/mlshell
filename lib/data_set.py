# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from time import time

from utils.mysql import MySQL


class DataSet(dict):

    def __init__(self):
        super(DataSet, self).__init__()
        self.metadata = self.get_metadata()
        self.query()

    def query(self, force=False):
        print('[INFO] Querying data:')
        for table, columns in self.metadata.items():
            ti = time()
            self[table] = MySQL.query("SELECT {} FROM {}".format(', '.join(columns), table))
            self[table].rename(columns={'id': '{}_id'.format(table[:-1])}, inplace=True)
            print('[INFO] > {} - {}s'.format(table, round(time() - ti, 2)))

    def get_metadata(self):
        return {
            'tabel1': [
                'col1',
                'col2',
                #'col3',
            ],
            'table2': [
                'col1',
                'meeting_idcol2',
                #'col3',
            ],
        }
