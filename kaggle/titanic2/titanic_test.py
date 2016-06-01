import unittest
import numpy.testing as npt
import pandas as pd

import titanic

AGE_MEDIAN = 28.0


class TestStuff(unittest.TestCase):
    def assert_preconditions(self, td):
        npt.assert_array_equal(
            ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
             'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
            td.columns,
            "Expected columns")

        self.assertTrue(
            td['Age'].isnull().values.sum() > 100,
            "has missing age values")
        self.assertEqual(
            AGE_MEDIAN,
            td['Age'].median())

        npt.assert_array_equal(
            ['male', 'female'],
            td['Sex'].unique(),
            "non numerical values for sex")

    def test_preconditions(self):
        td = pd.read_csv('train.csv')
        self.assert_preconditions(td)

    def test_preprocess(self):
        td = pd.read_csv('train.csv')
        preprocess_fn = titanic.make_preprocesser(td)

        td_preprocessed = preprocess_fn(td, scale=False)
        cabin_sectors = ["cabin_sector_{}".format(l) for l in 'bcde']
        embarked_hot = ["embarked_{}".format(l) for l in "CQS"]
        npt.assert_array_equal(
            ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'] + cabin_sectors + embarked_hot,
            td_preprocessed.columns,
            'Picked columns')

        # Age
        self.assertEqual(
            td_preprocessed['Age'].isnull().values.sum(),
            0,
            "no missing values")
        missing_age_idxs = td[['Age']].isnull().any(axis=1)
        npt.assert_array_equal(
            [AGE_MEDIAN],
            td_preprocessed[missing_age_idxs]['Age'].unique(),
            'previous NaN age values should be filled in by median')

        # Sex
        orig_sex = td['Sex'].head(10)
        mapped_sex = td_preprocessed['Sex'].head(10)
        npt.assert_array_equal(
            [{'male': 0, 'female': 1}[gender] for gender in orig_sex],
            mapped_sex,
            'should map male/female to 0/1')


        # make sure we haven't mutated the original
        self.assert_preconditions(td)

    def test_preprocess_with_scaling(self):
        td = pd.read_csv('train.csv')
        preprocess_fn = titanic.make_preprocesser(td)

        td_preprocessed = preprocess_fn(td, scale=True)

