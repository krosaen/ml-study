from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
import functools
import operator
from sklearn.pipeline import Pipeline, FeatureUnion


class BaseTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        return self


class ColumnExtractor(BaseTransformer):
    "Helps extract columns from Pandas Dataframe"

    def __init__(self, columns, c_type=None):
        self.columns = columns
        self.c_type = c_type

    def transform(self, X, **transform_params):
        cs = X[self.columns]
        if self.c_type is None:
            return cs
        else:
            return cs.astype(self.c_type)


class CategoricalColumnExtractor(BaseTransformer):

    def __init__(self, cat_vals):
        self.cat_vals = cat_vals

    def all_binary_cs(self):
        binary_cs = [['{}{}'.format(c, v) for v in vs] for c, vs in self.cat_vals.items()]
        return functools.reduce(operator.add, binary_cs)

    def transform(self, X, **transform_params):
        return X[self.all_binary_cs()]


class SpreadBinary(BaseTransformer):

    def transform(self, X, **transform_params):
        return X.applymap(lambda x: 1 if x == 1 else -1)


def dataframe_wrangler(*, quantitative_columns=None, categorical_columns=None):
    """
    Creates a Pipeline that will be ready to preprocess a dataframe.

    :param quantitative_columns: [q_column, ...]
    :param categorical_columns: {'cat_column': ['val1', 'val2', ...]}
    :return: Pipeline
    """
    if quantitative_columns is None:
        quantitative_columns = []
    if categorical_columns is None:
        categorical_columns = {}
    return Pipeline([
        ('features', FeatureUnion([
            ('quantitative', Pipeline([
                ('extract', ColumnExtractor(quantitative_columns, c_type=float)),
                ('scale', StandardScaler())
            ])),
            ('categorical', Pipeline([
                ('extract', CategoricalColumnExtractor(categorical_columns)),
                ('spread_binary', SpreadBinary())
            ])),
        ]))
    ])
