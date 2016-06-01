from sklearn.preprocessing import StandardScaler
import pandas as pd

def make_preprocesser(training_data):
    """
    Constructs a preprocessing function ready to apply to new dataframes.

    Crucially, the interpolating that is done based on the training data set
    is remembered so it can be applied to test datasets (e.g the mean age that
    is used to fill in missing values for 'Age' will be fixed based on the mean
    age within the training data set).

    Summary by column:

    ['PassengerId',
     'Survived',    # this is our target, not a feature
     'Pclass',      # keep as is: ordinal value should work, even though it's inverted (higher number is lower class cabin)
     'Name',        # omit (could try some fancy stuff like inferring ethnicity, but skip for now)
     'Sex',         # code to 0 / 1
     'Age',         # replace missing with median
     'SibSp',
     'Parch',
     'Ticket',      # omit (doesn't seem like low hanging fruit, could look more closely for pattern later)
     'Fare',        # keep, as fare could be finer grained proxy for socio economic status, sense of entitlement / power in getting on boat
     'Cabin',       # one hot encode using first letter as cabin as the cabin sector
     'Embarked']    # one hot encode

    Params:
        df: pandas.DataFrame containing the training data
    Returns:
        fn: a function to preprocess a dataframe (either before training or fitting a new dataset)
    """

    def pick_features(df):
        return df[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]

    # save median Age so we can use it to fill in missing data consistently
    # on any dataset
    median_age_series = training_data[['Age', 'Fare']].median()

    def fix_missing(df):
        return df.fillna(median_age_series)

    def map_sex(df):
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        return df

    def one_hot_cabin(df):
        def cabin_sector(cabin):
            if isinstance(cabin, str):
                return cabin[0].lower()
            else:
                return cabin

        df[['cabin_sector']] = df[['Cabin']].applymap(cabin_sector)
        one_hot = pd.get_dummies(df['cabin_sector'], prefix="cabin_sector")

        interesting_cabin_sectors = ["cabin_sector_{}".format(l) for l in 'bcde']

        for column, _ in one_hot.iteritems():
            if column.startswith('cabin_sector_') and column not in interesting_cabin_sectors:
                one_hot = one_hot.drop(column, axis=1)

        df = df.join(one_hot)

        df = df.drop('Cabin', axis=1)
        df = df.drop('cabin_sector', axis=1)
        return df

    def one_hot_embarked(df):
        one_hot = pd.get_dummies(df['Embarked'], prefix="embarked")
        df = df.join(one_hot)
        df = df.drop('Embarked', axis=1)
        return df

    # We want standard scaling fit on the training data, so we get a scaler ready
    # for application now. It needs to be applied to data that already has the other
    # pre-processing applied.
    training_data_all_but_scaled = map_sex(fix_missing(pick_features(training_data)))
    stdsc = StandardScaler()
    stdsc.fit(training_data_all_but_scaled[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])

    def scale_df(df):
        df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']] = \
            stdsc.transform(df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
        df[['Sex']] = df[['Sex']].applymap(lambda x: 1 if x == 1 else -1)
        for column, _ in df.iteritems():
            if column.startswith('cabin_sector_') or column.startswith('embarked_'):
                df[[column]] = df[[column]].applymap(lambda x: 1 if x == 1 else -1)
        return df

    def preprocess(df, scale=True):
        """
        Preprocesses a dataframe so it is ready for use with a model (either for training or prediction).

        Params:
            scale: whether to apply feature scaling. E.g with random forests feature scaling isn't necessary.
        """
        all_but_scaled = one_hot_embarked(one_hot_cabin(map_sex(fix_missing(pick_features(df)))))
        if scale:
            return scale_df(all_but_scaled)
        else:
            return all_but_scaled

    return preprocess
