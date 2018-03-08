
# coding: utf-8

import os
from datetime import datetime

import numpy as np
# np.random.seed(123)

import pandas as pd
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras import models

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import math
import lightgbm as lgb
import xgboost as xgb

from collections import Counter
from models import *

# # Read in data
a_train = pd.read_csv('../input/A_train.csv', index_col=['id'])
b_train = pd.read_csv('../input/B_train.csv', index_col=['id'])
c_train = pd.read_csv('../input/C_train.csv', index_col=['id'])
print(a_train.shape, b_train.shape, c_train.shape)

aX_train = a_train[list(set(a_train.columns.tolist()) - set(['poor']))]
bX_train = b_train[list(set(b_train.columns.tolist()) - set(['poor']))]
cX_train = c_train[list(set(c_train.columns.tolist()) - set(['poor']))]

ay_train = a_train['poor'].values
by_train = b_train['poor'].values
cy_train = c_train['poor'].values

print(aX_train.shape, bX_train.shape, cX_train.shape)

a_test = pd.read_csv('../input/A_test.csv', index_col=['id'])
b_test = pd.read_csv('../input/B_test.csv', index_col=['id'])
c_test = pd.read_csv('../input/C_test.csv', index_col=['id'])
print(a_test.shape, b_test.shape, c_test.shape)




# # Start training

paras_a = {
    'splits': 20,
    'lgb': {
        'max_depth': 4,
        'lr': 0.01,
        'hess': 3.,
        'feature_fraction': 0.07,
        'verbos_': 1000,
        'col_names': aX_train.columns.tolist(),
    },
    'xgb': {
        'eta': 0.01,
        'max_depth': 4,
        'subsample': 0.75,
        'colsample_by_tree': 0.07,
        'verbos_': 1000,
        'col_names': aX_train.columns.tolist(),
    },
    'use_nn': True,
    'nn': {
        'nn_l1': 300,
        'nn_l2': 300,
        'epochs': 75,
        'batch': 64,
        'dp': 0.,
    },
    'w_xgb': 0.45,
    'w_lgb': 0.25,
    'w_nn': 0.3,
}

a_preds, a_loss = train_model(aX_train, ay_train, paras_a, test_ = a_test)

paras_b = {
    'splits': 20,
    'lgb': {
        'max_depth': 3,
        'lr': 0.01,
        'hess': 3.,
        'feature_fraction': 0.025,
        'verbos_': 1000,
        'col_names': bX_train.columns.tolist(),
    },
    'xgb': {
        'eta': 0.01,
        'max_depth': 3,
        'subsample': 0.45,
        'colsample_by_tree': 0.03,
        'verbos_': 1000,
        'col_names': bX_train.columns.tolist(),
    },
    'use_nn': True,
    'nn': {
        'nn_l1': 400,
        'nn_l2': 400,
        'epochs': 30,
        'batch': 32,
        'dp': 0.25,
    },
    'w_xgb': 0.4,
    'w_lgb': 0.3,
    'w_nn': 0.3,
}

b_preds, b_loss = train_model(bX_train, by_train, paras_b, test_ = b_test)

used_features_c = [col for col in cX_train.columns if '_mean' not in col]
paras_c = {
    'splits': 10,
    'lgb': {
        'max_depth': 4,
        'lr': 0.01,
        'hess': 1.,
        'feature_fraction': 0.99,
        'verbos_': 1000,
        'col_names': used_features_c,
    },
    'xgb': {
        'eta': 0.01,
        'max_depth': 6,
        'subsample': 0.75,
        'colsample_by_tree': 0.75,
        'verbos_': 1000,
        'col_names': used_features_c,
    },
    'use_nn': False,
    'nn': {
        'nn_l1': 400,
        'nn_l2': 400,
        'epochs': 30,
        'batch': 32,
        'dp': 0.25,
    },
    'w_xgb': 0.7,
    'w_lgb': 0.3,
#     'w_nn': 0.3,
}

used_features_c = [col for col in cX_train.columns if '_mean' not in col]
c_preds, c_loss = train_model(cX_train[used_features_c], cy_train, paras_c, test_ = c_test[used_features_c])

def make_country_sub(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']
    
    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds,  # proba p=1
                               columns=['poor'], 
                               index=test_feat.index)
    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]

# convert preds to data frames
a_sub = make_country_sub(a_preds, a_test, 'A')
b_sub = make_country_sub(b_preds, b_test, 'B')
c_sub = make_country_sub(c_preds, c_test, 'C')
submission = pd.concat([a_sub, b_sub, c_sub])

print(a_sub.poor.agg(['max', 'min', 'mean', 'median']))
print(b_sub.poor.agg(['max', 'min', 'mean', 'median']))
print(c_sub.poor.agg(['max', 'min', 'mean', 'median']))

# lgb (new) 10 folds --------------------new
log_loss_mean = a_loss*(4041/8832) + b_loss*(1604/8832) + c_loss*(3187/8832)
print('Local logloss cross validation score: {}'.format(log_loss_mean))


submission.to_csv('../subs/sub_logloss_'+str(log_loss_mean)+'.csv')

