#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import warnings

import torch

torch.manual_seed(3407)
warnings.filterwarnings("ignore")
data = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA(4).csv')
# data = data.drop(columns=['Permeability','PAMPA'])
print(data.head())

# for i in data.columns:
#     if data[i].dtype=='object':
#         data.drop(columns=[i],inplace=True)
df = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA(4).csv')

y = df['Permeability'].values.reshape(-1,1)
# print(data)

df[df['Permeability'] >= -6] = 1
df[df['Permeability'] < -6] = 0
y = df['Permeability'].values
print(y)
y = y.astype('float32')


# from sklearn.preprocessing import scale
# df_x = scale(data)
# df_x = pd.DataFrame(df_x,columns=data.columns)




from sklearn.model_selection import KFold,train_test_split
# 创建10折交叉验证的对象
kf = KFold(n_splits=10, shuffle=True, random_state=3407)
i=1
# 循环遍历10次,训练和测试模型
X_stand = data.values
# print(X_stand)
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
    print('第{}折交叉验证数据保存成功'.format(i))
    i=i+1




