#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import warnings

import torch

torch.manual_seed(3407)
warnings.filterwarnings("ignore")
if __name__ == '__main__':

    data = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA(4).csv')

    df = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA(4).csv')

    y = df['Permeability'].values.reshape(-1,1)

    df[df['Permeability'] >= -6] = 1
    df[df['Permeability'] < -6] = 0
    y = df['Permeability'].values
    y = y.astype('float32')

    from sklearn.model_selection import KFold,train_test_split

    kf = KFold(n_splits=10, shuffle=True, random_state=3407)
    i=1
    X_stand = data.values
    for train_index, test_index in kf.split(X_stand):
        X_train, X_test = X_stand[train_index], X_stand[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          test_size=0.1,
                                                          random_state=3407)
        pd.DataFrame(X_train,columns=data.columns).to_csv('../data/data_splitClassifier/X_train{}.csv'.format(i),index=False)
        pd.DataFrame(X_test,columns=data.columns).to_csv('../data/data_splitClassifier/X_test{}.csv'.format(i),index=False)
        pd.DataFrame(X_val,columns=data.columns).to_csv('../data/data_splitClassifier/X_val{}.csv'.format(i),index=False)
        pd.DataFrame(y_train,columns=['target']).to_csv('../data/data_splitClassifier/y_train{}.csv'.format(i),index=False)
        pd.DataFrame(y_test,columns=['target']).to_csv('../data/data_splitClassifier/y_test{}.csv'.format(i),index=False)
        pd.DataFrame(y_val,columns=['target']).to_csv('../data/data_splitClassifier/y_val{}.csv'.format(i),index=False)
        print('experiment_{}_data_save'.format(i))
        i=i+1




