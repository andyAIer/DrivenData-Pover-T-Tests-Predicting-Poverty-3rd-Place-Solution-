
# coding: utf-8

import os

import numpy as np
import pandas as pd

from datetime import datetime
from collections import Counter


# # Start for feature processing

def merge_add_features(train, test):
    
    merge = pd.concat([train.drop('poor', axis=1), test], axis=0)
    df_new = pd.DataFrame(data=merge.id.unique(), columns=['id'])

    cat_ = []
    num_ = []
    for col in train.columns:
        if train[col].dtype in ['int64', 'float64'] and col not in ['id', 'iid']:
            num_.append(col)
#             continue
        elif train[col].dtype=='O' and col not in ['poor', 'country']:
            cat_.append(col)
    print("Merged table shape: ", merge.shape, 'Categorical features\', number: ', len(cat_), "Numerical features' number: ", len(num_))

    ids = df_new.id.tolist()
    len_ = len(ids)
    print('number of id: ', len_)
    
    for col in cat_:
        unique_ = merge[col][merge[col].notnull()].unique()
        for v in unique_:
            df_new[col+'_'+v] = 0
        
    for idx in range(len_):
        if idx % 1000 == 0:
            print(idx, ids[idx], str(datetime.now()))
        for col in cat_:
            ct = Counter(merge[merge.id==ids[idx]][col].tolist())
            for k in ct:
                df_new.at[idx, col+'_'+k] += ct[k]

    for col in num_:
        df_new[col+'_mean'] = np.NaN
        df_new[col+'_max'] = np.NaN
        df_new[col+'_min'] = np.NaN
        df_new[col+'_pos'] = 0.
        df_new[col+'_neg'] = 0.
    for idx, id_ in enumerate(merge.id.unique()):
        if idx % 500 == 0:
            print(idx, id_, str(datetime.now()))
        df_new.at[df_new.id==id_, 'family_num'] = merge[merge.id==id_].shape[0]
        for col in num_:
            li = merge[merge.id==id_][col]
            df_new.at[df_new.id==id_, col+'_mean'] = li.mean()
            df_new.at[df_new.id==id_, col+'_max'] = li.max()
            df_new.at[df_new.id==id_, col+'_min'] = li.min()
            df_new.at[df_new.id==id_, col+'_pos'] = len([x for x in li.tolist() if x > 0]) 
            df_new.at[df_new.id==id_, col+'_neg'] = len([x for x in li.tolist() if x < 0]) 
    print("Finish, shape of joined table: ", df_new.shape)
    return df_new

ai_train = pd.read_csv('../input/A_indiv_train.csv')
ai_test = pd.read_csv('../input/A_indiv_test.csv')

bi_train = pd.read_csv('../input/B_indiv_train.csv')
bi_test = pd.read_csv('../input/B_indiv_test.csv')

ci_train = pd.read_csv('../input/C_indiv_train.csv')
ci_test = pd.read_csv('../input/C_indiv_test.csv')

df_new_a = merge_add_features(ai_train, ai_test)
df_new_b = merge_add_features(bi_train, bi_test)
df_new_c = merge_add_features(ci_train, ci_test)

df_new_b[:len(bi_train.id.unique())].to_csv('../input/feature_b_train_ind.csv', index=False)
df_new_b[len(bi_train.id.unique()):].to_csv('../input/feature_b_test_ind.csv', index=False)

df_new_a[:len(ai_train.id.unique())].to_csv('../input/feature_a_train_ind.csv', index=False)
df_new_a[len(ai_train.id.unique()):].to_csv('../input/feature_a_test_ind.csv', index=False)

df_new_c[:len(ci_train.id.unique())].to_csv('../input/feature_c_train_ind.csv', index=False)
df_new_c[len(ci_train.id.unique()):].to_csv('../input/feature_c_test_ind.csv', index=False)

