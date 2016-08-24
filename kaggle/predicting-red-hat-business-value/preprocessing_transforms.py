from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd


class BaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        return self


class ColumnSelector(BaseTransformer):
    """Selects columns from Pandas Dataframe"""

    def __init__(self, columns, c_type=None):
        self.columns = columns
        self.c_type = c_type

    def transform(self, X, **transform_params):
        cs = X[self.columns]
        if self.c_type is None:
            return cs
        else:
            return cs.astype(self.c_type)


class SpreadBinary(BaseTransformer):

    def transform(self, X, **transform_params):
        return X.applymap(lambda x: 1 if x == 1 else -1)


class DfTransformerAdapter(BaseTransformer):
    """Adapts a scikit-learn Transformer to return a pandas DataFrame"""

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None, **fit_params):
        self.transformer.fit(X, y=y, **fit_params)
        return self

    def transform(self, X, **transform_params):
        raw_result = self.transformer.transform(X, **transform_params)
        return pd.DataFrame(raw_result, columns=X.columns, index=X.index)


class DfOneHot(BaseTransformer):
    """
    Wraps helper method `get_dummies` making sure all columns get one-hot encoded.
    """
    def __init__(self):
        self.dummy_columns = []

    def fit(self, X, y=None, **fit_params):
        self.dummy_columns = pd.get_dummies(
            X,
            prefix=[c for c in X.columns],
            columns=X.columns).columns
        return self

    def transform(self, X, **transform_params):
        return pd.get_dummies(
            X,
            prefix=[c for c in X.columns],
            columns=X.columns).reindex(columns=self.dummy_columns, fill_value=0)


class DfFeatureUnion(BaseTransformer):
    """A dataframe friendly implementation of `FeatureUnion`"""

    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, X, y=None, **fit_params):
        for l, t in self.transformers:
            t.fit(X, y=y, **fit_params)
        return self

    def transform(self, X, **transform_params):
        transform_results = [t.transform(X, **transform_params) for l, t in self.transformers]
        return pd.concat(transform_results, axis=1)
