from sklearn.preprocessing import StandardScaler


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
     'Cabin',       # omit (10% are missing values, could look more closely later, one idea would be to break this out into a few different boolean variables for major section, A->E)
     'Embarked']    # omit (could one-hot encode it, but can't see how this would affect survivorship, let's be lazy to start)

    Params:
        df: pandas.DataFrame containing the training data
    Returns:
        fn: a function to preprocess a dataframe (either before training or fitting a new dataset)
    """

    def pick_features(df):
        return df[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

    # save median Age so we can use it to fill in missing data consistently
    # on any dataset
    median_age_series = training_data[['Age', 'Fare']].median()

    def fix_missing(df):
        return df.fillna(median_age_series)

    def map_categorical(df):
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        return df

    # We want standard scaling fit on the training data, so we get a scaler ready
    # for application now. It needs to be applied to data that already has the other
    # pre-processing applied.
    training_data_all_but_scaled = map_categorical(fix_missing(pick_features(training_data)))
    stdsc = StandardScaler()
    stdsc.fit(training_data_all_but_scaled[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])

    def scale_df(df):
        df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']] = \
            stdsc.transform(df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
        df[['Sex']] = df[['Sex']].applymap(lambda x: 1 if x == 1 else -1)
        return df

    def preprocess(df, scale=True):
        """
        Preprocesses a dataframe so it is ready for use with a model (either for training or prediction).

        Params:
            scale: whether to apply feature scaling. E.g with random forests feature scaling isn't necessary.
        """
        all_but_scaled = map_categorical(fix_missing(pick_features(df)))
        if scale:
            return scale_df(all_but_scaled)
        else:
            return all_but_scaled

    return preprocess
