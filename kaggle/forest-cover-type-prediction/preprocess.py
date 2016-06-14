from sklearn.preprocessing import StandardScaler
import functools
import operator


def make_preprocessor(td, column_summary):
    # it's important to scale consistently on all preprocessing based on
    # consistent scaling, so we do it once and keep ahold of it for all future
    # scaling.
    stdsc = StandardScaler()
    stdsc.fit(td[column_summary['quantitative']])

    def scale_q(df, column_summary):
        df[column_summary['quantitative']] = stdsc.transform(df[column_summary['quantitative']])
        return df, column_summary

    def scale_binary_c(df, column_summary):
        binary_cs = [['{}{}'.format(c, v) for v in vs] for c, vs in column_summary['categorical'].items()]
        all_binary_cs = functools.reduce(operator.add, binary_cs)
        df[all_binary_cs] = df[all_binary_cs].applymap(lambda x: 1 if x == 1 else -1)
        return df, column_summary

    def preprocess(df):
        fns = [scale_q, scale_binary_c]

        cs = column_summary
        for fn in fns:
            df, cs = fn(df, cs)

        return df

    return preprocess, column_summary
