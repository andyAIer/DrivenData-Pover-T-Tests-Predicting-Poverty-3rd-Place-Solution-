
# coding: utf-8

import os
from datetime import datetime

import numpy as np
# np.random.seed(123)
import pandas as pd

# from collections import Counter

# data directory
DATA_DIR = os.path.join('..', 'input')

# load *_hhold training data
a_train = pd.read_csv('../input/A_hhold_train.csv', index_col='id')
b_train = pd.read_csv('../input/B_hhold_train.csv', index_col='id')
c_train = pd.read_csv('../input/C_hhold_train.csv', index_col='id')
# load test data
a_test_o = pd.read_csv('../input/A_hhold_test.csv', index_col='id')
b_test_o = pd.read_csv('../input/B_hhold_test.csv', index_col='id')
c_test_o = pd.read_csv('../input/C_hhold_test.csv', index_col='id')

a_feature_1 = pd.read_csv('../input/feature_a_train_ind.csv', index_col=['id'])
b_feature_1 = pd.read_csv('../input/feature_b_train_ind.csv', index_col=['id'])
c_feature_1 = pd.read_csv('../input/feature_c_train_ind.csv', index_col=['id'])

a_feature_1_test = pd.read_csv('../input/feature_a_test_ind.csv', index_col=['id'])
b_feature_1_test = pd.read_csv('../input/feature_b_test_ind.csv', index_col=['id'])
c_feature_1_test = pd.read_csv('../input/feature_c_test_ind.csv', index_col=['id'])

print('Shape of hhold data A, B, C, Train and test: \n', a_train.shape, b_train.shape, c_train.shape, a_test_o.shape, b_test_o.shape, c_test_o.shape)
print('Used features\' shape from individuals for A, B, C (train, test): \n', a_feature_1.shape, b_feature_1.shape, c_feature_1.shape, a_feature_1_test.shape, b_feature_1_test.shape, c_feature_1_test.shape)
print('Mean value of "poor" for A, B, C: \n',a_train.poor.mean(), b_train.poor.mean(), c_train.poor.mean())

# join *_hhold data with the data from individual data data tables
a_used_features = [col for col in a_feature_1.columns if '_mean' in col or col == 'family_num']
b_used_features = [col for col in b_feature_1.columns if '_mean' in col or col == 'family_num']
c_used_features = [col for col in c_feature_1.columns if col == 'family_num']

a_feature_1 = a_feature_1[a_used_features]
b_feature_1 = b_feature_1[b_used_features]
c_feature_1 = c_feature_1[c_used_features]

a_feature_1_test = a_feature_1_test[a_used_features]
b_feature_1_test = b_feature_1_test[b_used_features]
c_feature_1_test = c_feature_1_test[c_used_features]

a_train = a_train.join(a_feature_1, how='inner')
b_train = b_train.join(b_feature_1, how='inner')
c_train = c_train.join(c_feature_1, how='inner')

a_test_o = a_test_o.join(a_feature_1_test, how='inner')
b_test_o = b_test_o.join(b_feature_1_test, how='inner')
c_test_o = c_test_o.join(c_feature_1_test, how='inner')
print(a_train.shape, b_train.shape, c_train.shape)

num_cols_a = [col for col in a_train.columns if a_train[col].dtype in ['int64', 'float64'] and col not in ['id']]
# print(num_cols_a)
for col in num_cols_a:
    a_train[col + '_ave'] = a_train[col]/a_train['family_num']
    a_test_o[col + '_ave'] = a_test_o[col]/a_test_o['family_num']
    
# a_hhold = pd.concat([a_train, a_test_o], axis=0)
num_cols_b = [col for col in b_train.columns if b_train[col].dtype in ['int64', 'float64'] and col not in ['id']]
for col in num_cols_b:
    b_train[col + '_ave'] = b_train[col]/b_train['family_num']
    b_test_o[col + '_ave'] = b_test_o[col]/b_test_o['family_num']
    
num_cols_c = [col for col in c_train.columns if c_train[col].dtype in ['int64', 'float64'] and col not in ['id']]

for col in num_cols_c:
    c_train[col + '_ave'] = c_train[col]/c_train['family_num']
    c_test_o[col + '_ave'] = c_test_o[col]/c_test_o['family_num']
a_hhold = pd.concat([a_train.drop(['country'], axis=1), a_test_o], axis=0)
# a_hhold = pd.concat([a_train, a_test_o], axis=0)
b_hhold = pd.concat([b_train.drop(['country'], axis=1), b_test_o], axis=0)
c_hhold = pd.concat([c_train.drop(['country'], axis=1), c_test_o], axis=0)
print(a_hhold.shape, b_hhold.shape, c_hhold.shape)
# Standardize features
def standardize(df, numeric_only=True):
    
    numeric = df.select_dtypes(include=['int64', 'float64'])
    # subtracy mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()
    return df

# def encode_cat(df):
#     for col in df.columns:
#         if df[col].dtype not in ['int64', 'float64', 'bool']:
#             len_ = df[col].unique()
#             dict_ = {cl: i for i, cl in enumerate(df[col].unique())}
# #             df[col] = df[col].astype('category')
#             df[col] = df[col].apply(lambda x: dict_[x])
#     return df
    
def pre_process_data(df, nn=False):
    print("Input shape:\t{}".format(df.shape))
    df = standardize(df)
#     print("After standardization {}".format(df.shape))
    if nn:
        current_cols = df.columns
#         print('poor' in df.columns)
        poor_ = df.poor.values
        df = pd.get_dummies(df.drop('poor', axis=1), drop_first=True)
        df['poor'] = poor_.astype('bool')
        all_nan_cols = []
        for col in df.columns:
            if df[col].isnull().sum() == df.shape[0]:
                all_nan_cols.append(col)
        df.drop(all_nan_cols, axis=1, inplace=True)
        df.fillna(df.median(), inplace=True)
    else:
        df = encode_cat(df)
    print('Final shape {}'.format(df.shape))
    return df

a_hhold_p = pre_process_data(a_hhold, nn=True)
print()
b_hhold_p = pre_process_data(b_hhold, nn=True)
print()
c_hhold_p = pre_process_data(c_hhold, nn=True)

aX_train = a_hhold_p.drop('poor', axis=1)[:a_train.shape[0]]
ay_train = np.ravel(a_hhold_p.poor)[:a_train.shape[0]]
a_test = a_hhold_p.drop('poor', axis=1)[a_train.shape[0]:]

bX_train = b_hhold_p.drop('poor', axis=1)[:b_train.shape[0]]
by_train = np.ravel(b_hhold_p.poor)[:b_train.shape[0]]
b_test = b_hhold_p.drop('poor', axis=1)[b_train.shape[0]:]

cX_train = c_hhold_p.drop('poor', axis=1)[:c_train.shape[0]]
cy_train = np.ravel(c_hhold_p.poor)[:c_train.shape[0]]
c_test = c_hhold_p.drop('poor', axis=1)[c_train.shape[0]:]

aX_train['poor'] = ay_train
bX_train['poor'] = by_train
cX_train['poor'] = cy_train

aX_train.to_csv('../input/A_train.csv')
bX_train.to_csv('../input/B_train.csv')
cX_train.to_csv('../input/C_train.csv')

a_test.to_csv('../input/A_test.csv')
b_test.to_csv('../input/B_test.csv')
c_test.to_csv('../input/C_test.csv')

