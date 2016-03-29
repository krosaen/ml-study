import unittest
import numpy as np
from adeline import *
import pprint
import pandas as pd
import itertools

class AdelineTestCase(unittest.TestCase):
    def test_the_thing(self):


        df = pd.read_csv('iris.data', header=None)
        sepal_petal_rows = df.iloc[0:100, [0, 2]].values
        setosa_rows = sepal_petal_rows[:50]
        versicolor_rows = sepal_petal_rows[50:]
        y = df.iloc[0:100, 4].values
        y = np.where(y == 'Iris-setosa', -1, 1)

        predict, weights_log, raw_outputs_log, errors_log, num_errors_log = train_adeline(sepal_petal_rows, y)
        print("ys")
        pprint.pprint(y)
        print("\n\nlog\n\n")
        # pprint.pprint(list(zip(weights_log, raw_outputs_log, errors_log, num_errors_log)))
        pprint.pprint(list(zip(weights_log, num_errors_log)))
        self.assertEqual(
            [[], [], []],
            list(zip(weights_log, errors_log, num_errors_log)))


