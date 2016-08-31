from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import heapq
import numpy as np

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


class OmniEncoder(BaseTransformer):
    """
    Encodes a categorical variable using no more than k columns. As many values as possible
    are one-hot encoded, the remaining are fit within a binary encoded set of columns.
    If necessary some are dropped (e.g if (#unique_values) > 2^k).

    In deciding which values to one-hot encode, those that appear more frequently are
    preferred.
    """
    def __init__(self, max_cols=20):
        self.column_infos = {}
        self.max_cols = max_cols
        if max_cols < 3 or max_cols > 100:
            raise ValueError("max_cols {} not within range(3, 100)".format(max_cols))

    def fit(self, X, y=None, **fit_params):
        self.column_infos = {col: self._column_info(X[col], self.max_cols) for col in X.columns}
        return self

    def transform(self, X, **transform_params):
        return pd.concat(
            [self._encode_column(X[col], self.max_cols, *self.column_infos[col]) for col in X.columns],
            axis=1
        )

    @staticmethod
    def _encode_column(col, max_cols, one_hot_vals, binary_encoded_vals):
        num_one_hot = len(one_hot_vals)
        num_bits = max_cols - num_one_hot if len(binary_encoded_vals) > 0 else 0

        # http://stackoverflow.com/a/29091970/231589
        zero_base = ord('0')
        def i_to_bit_array(i):
            return np.fromstring(
                    np.binary_repr(i, width=num_bits),
                    'u1'
                ) - zero_base

        binary_val_to_bit_array = {val: i_to_bit_array(idx + 1) for idx, val in enumerate(binary_encoded_vals)}

        bit_cols = [np.binary_repr(2 ** i, width=num_bits) for i in reversed(range(num_bits))]

        col_names = ["{}_{}".format(col.name, val) for val in one_hot_vals] + ["{}_{}".format(col.name, bit_col) for bit_col in bit_cols]

        zero_bits = np.zeros(num_bits, dtype=np.int)

        def splat(v):
            v_one_hot = [1 if v == ohv else 0 for ohv in one_hot_vals]
            v_bits = binary_val_to_bit_array.get(v, zero_bits)

            return pd.Series(np.concatenate([v_one_hot, v_bits]))

        df = col.apply(splat)
        df.columns = col_names

        return df

    @staticmethod
    def _column_info(col, max_cols):
        """

        :param col: pd.Series
        :return: {'val': 44, 'val2': 4, ...}
        """
        val_counts = dict(col.value_counts())
        num_one_hot = OmniEncoder._num_onehot(len(val_counts), max_cols)
        return OmniEncoder._partition_one_hot(val_counts, num_one_hot)

    @staticmethod
    def _partition_one_hot(val_counts, num_one_hot):
        """
        Paritions the values in val counts into a list of values that should be
        one-hot encoded and a list of values that should be binary encoded.

        The `num_one_hot` most popular values are chosen to be one-hot encoded.

        :param val_counts: {'val': 433}
        :param num_one_hot: the number of elements to be one-hot encoded
        :return: ['val1', 'val2'], ['val55', 'val59']
        """
        one_hot_vals = [k for (k, count) in heapq.nlargest(num_one_hot, val_counts.items(), key=lambda t: t[1])]
        one_hot_vals_lookup = set(one_hot_vals)

        bin_encoded_vals = [val for val in val_counts if val not in one_hot_vals_lookup]

        return sorted(one_hot_vals), sorted(bin_encoded_vals)


    @staticmethod
    def _num_onehot(n, k):
        """
        Determines the number of onehot columns we can have to encode n values
        in no more than k columns, assuming we will binary encode the rest.

        :param n: The number of unique values to encode
        :param k: The maximum number of columns we have
        :return: The number of one-hot columns to use
        """
        num_one_hot = min(n, k)

        def num_bin_vals(num):
            if num == 0:
                return 0
            return 2 ** num - 1

        def capacity(oh):
            """
            Capacity given we are using `oh` one hot columns.
            """
            return oh + num_bin_vals(k - oh)

        while capacity(num_one_hot) < n and num_one_hot > 0:
            num_one_hot -= 1

        return num_one_hot


class EncodeCategorical(BaseTransformer):
    def __init__(self):
        self.categorical_vals = {}

    def fit(self, X, y=None, **fit_params):
        self.categorical_vals = {col: {label: idx + 1 for idx, label in enumerate(sorted(X[col].dropna().unique()))} for
                                 col in X.columns}
        return self

    def transform(self, X, **transform_params):
        return pd.concat(
            [X[col].map(self.categorical_vals[col]) for col in X.columns],
            axis=1
        )


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
