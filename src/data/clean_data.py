import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

np.random.seed(42)

def clean_dataframes(train, test):
    '''Runs the different steps in the cleaning process.
    Input: 2 pandas dataframes
    Output: 2 pandas dataframes'''

    train, test = replace_illogical(train, test)
    train, test = lowercase_features(train, test)
    train, test = fill_nans(train, test)
    train, test = dummy_installer(train, test)
    train, test = fill_construction_year(train, test)
    train, test = make_naive_bayes(train, test)
    return train, test

def replace_illogical(train, test):
    '''Takes in the training and testing dataframes and cleans illogical values from feature folumns.
    Input: 2 pandas dataframes
    Output: 2 pandas dataframes'''

    for df in [train, test]:
        df['funder'].replace('0', np.nan, inplace=True)
        df['installer'].replace('0', np.nan, inplace=True)
        df['longitude'].replace(0, np.nan, inplace=True)
        df['latitude'].replace(-0.00000002, np.nan, inplace=True)
        df['population'].replace(0, np.nan, inplace=True)
        df['construction_year'].replace(0, np.nan, inplace=True)
    return train, test

def lowercase_features(train, test):
    '''Takes in the training and testing dataframes and makes all text lowercase to ensure values that are meant to be the same are recognized as such.
    Input: 2 pandas dataframes
    Output: 2 pandas dataframes'''

    for df in [train, test]:
        for col in df:
            if df[col].dtype == 'O' and col not in ['permit', 'public_meeting']:
                df[col] = df[col].str.lower()
    return train, test

def fill_nans(train, test):
    '''Fills the nans in the easier to fill columns.
    Input: 2 pandas dataframes
    Output: 2 pandas dataframes'''

    train['public_meeting'].fillna(value=True, inplace=True)
    test['public_meeting'].fillna(value=True, inplace=True)
    train['permit'].fillna(value=True, inplace=True)
    test['permit'].fillna(value=True, inplace=True)
    train['longitude'].fillna(value=train['longitude'].mean(), inplace=True)
    test['longitude'].fillna(value=test['longitude'].mean(), inplace=True)
    train['latitude'].fillna(value=train['latitude'].mean(), inplace=True)
    test['latitude'].fillna(value=test['latitude'].mean(), inplace=True)
    train['population'].fillna(value=train['population'].mean(), inplace=True)
    test['population'].fillna(value=train['population'].mean(), inplace=True)
    return train, test

def dummy_installer(train, test):
    '''Makes dummy columns for the top 10 installers by number of installations.
    Input: 2 pandas dataframes
    Output: 2 pandas dataframes'''

    dummy_index = train['installer'].value_counts()[:10].index.values
    dummy_df_train = pd.get_dummies(train['installer'])[dummy_index]
    dummy_df_test = pd.get_dummies(test['installer'])[dummy_index]
    train = pd.concat([train, dummy_df_train], axis=1)
    test = pd.concat([test, dummy_df_test], axis=1)
    return train, test

def fill_construction_year(train, test):
    '''Fills nans in construction year column by creating a random forest regressor with construction year as the target.
    Input: 2 pandas dataframes
    Output: 2 pandas dataframes'''

    dummies_train = pd.get_dummies(train[['basin', 'extraction_type_class', 'management', 'water_quality', 'quantity', 'source']])
    train = pd.concat([train, dummies_train], axis=1)
    clean_constr_train = train.drop(['funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region', 'lga', 'ward', 'scheme_management', 'scheme_name', 'extraction_type', 'management', 'management_group', 'payment', 'payment_type', 'water_quality', 'quality_group', 'quantity'\
     'source', 'source_type', 'waterpoint_type', 'recorded_by', 'extraction_type_group', 'extraction_type_class', 'quantity_group', 'source_class', 'waterpoint_type_group', 'region_code', 'district_code', 'status_group', 'id', 'population'], axis=1)

    dummies_test = pd.get_dummies(test[['basin', 'extraction_type_class', 'management', 'water_quality', 'quantity', 'source']])
    test = pd.concat([test, dummies_test], axis=1)
    clean_constr_test = test.drop(['funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region', 'lga', 'ward', 'scheme_management', 'scheme_name', 'extraction_type', 'management', 'management_group', 'payment', 'payment_type', 'water_quality', 'quality_group', 'quantity',\
     'source', 'source_type', 'waterpoint_type', 'recorded_by', 'extraction_type_group', 'extraction_type_class', 'quantity_group', 'source_class', 'waterpoint_type_group', 'region_code', 'district_code', 'id', 'population'], axis=1)

    constr_y_train = clean_constr_train[~clean_constr_train['construction_year'].isnull()]['construction_year']
    constr_y_test = clean_constr_test[~clean_constr_test['construction_year'].isnull()]['construction_year']
    constr_y_total = pd.concat([constr_y_train, constr_y_test], axis=0, ignore_index=True).values

    constr_X_train = clean_constr_train[~clean_constr_train['construction_year'].isnull()][['amount_tsh', 'gps_height', 'longitude', 'latitude', 'num_private', 'public_meeting', 'permit']]
    constr_X_test = clean_constr_test[~clean_constr_test['construction_year'].isnull()][['amount_tsh', 'gps_height', 'longitude', 'latitude', 'num_private', 'public_meeting', 'permit']]
    constr_X_total = pd.concat([constr_X_train, constr_X_test], axis=0, ignore_index=True).values

    constr_X_train, constr_X_test, constr_y_train, constr_y_test = train_test_split(constr_X_total, constr_y_total)

    model_constr = RandomForestRegressor()
    model_constr.fit(constr_X_train, constr_y_train)
    mse = mean_squared_error(constr_y_test, model_constr.predict(constr_X_test))
    # print np.sqrt(mse)

    prediction_df_train = train[train['construction_year'].isnull()]
    train_preds = model_constr.predict(prediction_df_train[['amount_tsh', 'gps_height', 'longitude', 'latitude', 'num_private', 'public_meeting', 'permit']].values)
    train.ix[train['construction_year'].isnull(), 'construction_year'] = train_preds

    prediction_df_test = test[test['construction_year'].isnull()]
    test_preds = model_constr.predict(prediction_df_test[['amount_tsh', 'gps_height', 'longitude', 'latitude', 'num_private', 'public_meeting', 'permit']].values)
    test.ix[test['construction_year'].isnull(), 'construction_year'] = test_preds

    return train, test

def tokenize_snow(doc):
    snowball = SnowballStemmer('english')
    return [snowball.stem(word) for word in word_tokenize(doc.lower())]

def make_model(tokenize_func, x_train, y_train):
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize_func)
    tfidf_vectorized = vectorizer.fit_transform(x_train)
    model = MultinomialNB()
    model.fit(tfidf_vectorized, y_train)
    return model, vectorizer

def make_naive_bayes(train, test):
    '''Aggregates the text columns in the dataframe and creates a naive bayes classifier using the text as the input.  Naive bayes predictions then become a new feature.
    Input: 2 pandas dataframes
    Output: 2 pandas dataframes'''

    train['text_cols'] = train[['funder', 'wpt_name', 'subvillage', 'region', 'lga', 'ward', 'scheme_management', 'scheme_name', 'extraction_type', 'management_group', 'quality_group', 'waterpoint_type']].values.tolist()
    train.drop(['funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region', 'lga', 'ward', 'scheme_management', 'scheme_name', 'extraction_type', 'management', 'management_group', 'payment', 'payment_type', 'water_quality', 'quality_group', 'quantity', 'source',\
     'source_type', 'waterpoint_type', 'recorded_by', 'extraction_type_group', 'extraction_type_class', 'quantity_group', 'source_class', 'waterpoint_type_group', 'region_code', 'district_code'], axis=1, inplace=True)

    test['text_cols'] = test[['funder', 'wpt_name', 'subvillage', 'region', 'lga', 'ward', 'scheme_management', 'scheme_name', 'extraction_type', 'management_group', 'quality_group', 'waterpoint_type']].values.tolist()
    test.drop(['funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region', 'lga', 'ward', 'scheme_management', 'scheme_name', 'extraction_type', 'management', 'management_group', 'payment', 'payment_type', 'water_quality', 'quality_group', 'quantity', 'source',\
     'source_type', 'waterpoint_type', 'recorded_by', 'extraction_type_group', 'extraction_type_class', 'quantity_group', 'source_class', 'waterpoint_type_group', 'region_code', 'district_code'], axis=1, inplace=True)

    train['text_cols'] = train['text_cols'].map(lambda x: ' '.join(str(word) for word in x))
    test['text_cols'] = test['text_cols'].map(lambda x: ' '.join(str(word) for word in x))

    naive_y = train['status_group'].values
    naive_X = train['text_cols'].values
    naive_X_train, naive_X_test, naive_y_train, naive_y_test = train_test_split(naive_X, naive_y, stratify=naive_y)

    model_snow, vect_snow = make_model(tokenize_snow, naive_X_train, naive_y_train)

    train['naive_bayes'] = model_snow.predict(vect_snow.transform(naive_X))
    test['naive_bayes'] = model_snow.predict(vect_snow.transform(test['text_cols'].values))

    train.drop('text_cols', axis=1, inplace=True)
    test.drop('text_cols', axis=1, inplace=True)
    naive_dummies_train = pd.get_dummies(train['naive_bayes'], drop_first=True)
    naive_dummies_test = pd.get_dummies(test['naive_bayes'], drop_first=True)

    train['non functional'] = 0
    train['functional needs repair'] = 0
    test['non functional'] = 0
    test['functional needs repair'] = 0

    for col in naive_dummies_train.columns:
        train[col] = naive_dummies_train[col]
    for col in naive_dummies_test.columns:
        test[col] = naive_dummies_test[col]

    train.drop('naive_bayes', axis=1, inplace=True)
    test.drop('naive_bayes', axis=1, inplace=True)

    return train, test

if __name__ == '__main__':
    train_values = pd.read_csv('data/raw/training_values.csv')
    train_labels = pd.read_csv('data/raw/training_labels.csv')
    train_df = train_values.merge(train_labels, on='id')

    test_values = pd.read_csv('data/raw/test_values.csv')
    clean_train = train_df
    clean_test = test_values
    clean_train, clean_test = clean_dataframes(clean_train, clean_test)
    clean_train.to_csv('data/processed/clean_train_dummies.csv')
    clean_test.to_csv('data/processed/clean_test_dummies.csv')
