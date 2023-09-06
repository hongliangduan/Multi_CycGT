#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA(4).csv')




from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def mol_to_vector(data):
    arrs = np.zeros((data.shape[0],128))
    for i in range(data.shape[0]):
        smiles = data[i]
        mol = Chem.MolFromSmiles(smiles)
        # 将mol对象转化为分子指纹
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2,nBits=128)

        # 将指纹转化为numpy数组并返回
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        arrs[i] = arr
    return arrs
arrs = mol_to_vector(data['SMILES'].values)
columns_list = ['SMILES{}'.format(i+1) for i in range(arrs.shape[1])]

for i in data.columns:
    if data[i].dtype=='object':
        data.drop(columns=[i],inplace=True)
data.drop(columns=['PAMPA'],inplace=True)
y = data['Permeability'].values.reshape(-1,1)
X = data.drop(columns=['Permeability'])
from sklearn.preprocessing import scale
X_stand = scale(X)
data[data['Permeability']>=-6]=1
data[data['Permeability']<-6]=0
y = data['Permeability'].values
y = y.astype('float32')
df_X = pd.DataFrame(X_stand,columns=X.columns)
df_X[columns_list] = arrs



df = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA(4).csv')

df_X = df_X.drop(columns=['CycPeptMPDB_ID','Year'])
# df_X['SMILES'] = df['SMILES'].values
print(df_X.shape)
# exit()
from sklearn.model_selection import KFold,train_test_split
# 创建10折交叉验证的对象
kf = KFold(n_splits=10, shuffle=True, random_state=3407)
i=1
# 循环遍历10次,训练和测试模型
X_stand = df_X.values
for train_index, test_index in kf.split(X_stand):
    X_train, X_test = X_stand[train_index], X_stand[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.1,
                                                      random_state=3407)
    pd.DataFrame(X_train,columns=df_X.columns).to_csv('../data/data_splitClassifier_362/X_train{}.csv'.format(i),index=False)
    pd.DataFrame(X_test,columns=df_X.columns).to_csv('../data/data_splitClassifier_362/X_test{}.csv'.format(i),index=False)
    pd.DataFrame(X_val,columns=df_X.columns).to_csv('../data/data_splitClassifier_362/X_val{}.csv'.format(i),index=False)
    pd.DataFrame(y_train,columns=['target']).to_csv('../data/data_splitClassifier_362/y_train{}.csv'.format(i),index=False)
    pd.DataFrame(y_test,columns=['target']).to_csv('../data/data_splitClassifier_362/y_test{}.csv'.format(i),index=False)
    pd.DataFrame(y_val,columns=['target']).to_csv('../data/data_splitClassifier_362/y_val{}.csv'.format(i),index=False)
    print('第{}折交叉验证数据保存成功'.format(i))
    i=i+1


print(df_X.shape)
