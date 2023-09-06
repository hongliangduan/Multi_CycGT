#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import scale
import torch

torch.manual_seed(3407)
warnings.filterwarnings("ignore")
data = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA(4).csv')
data = data.drop(columns=['Permeability','PAMPA'])
print(data.head())

for i in data.columns:
    if data[i].dtype=='object':
        data.drop(columns=[i],inplace=True)
df = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA(4).csv')
data_scale = scale(data)
data = pd.DataFrame(data_scale,columns=data.columns)

data['SMILES'] = df['SMILES']
data['Sequence_LogP'] = df['Sequence_LogP']
data['Sequence_TPSA'] = df['Sequence_TPSA']

# y = df['Permeability'].values.reshape(-1,1)
# print(data)
#
# df[df['Permeability'] >= -6] = 1
# df[df['Permeability'] < -6] = 0
# y = df['Permeability']
# y = y.astype('float32')

def fenlei(x):
    if x >=-6:
        return 1
    else:
        return 0


data['label']=df['Permeability'].apply(fenlei)
# data['label'] = y
data['seq_len'] = df['Monomer_Length']


from sklearn.model_selection import KFold,train_test_split
# 创建10折交叉验证的对象
kf = KFold(n_splits=10, shuffle=True, random_state=3407)
i=1

# 循环遍历10次,训练和测试模型
X_stand = data.values

print(X_stand.shape)
# print(X_stand)
for train_index, test_index in kf.split(X_stand):
    X_train, X_test = X_stand[train_index], X_stand[test_index]
    # y_train, y_test = y[train_index], y[test_index]
    X_train, X_val, y_train, y_val = train_test_split(X_train, X_train,
                                                      test_size=0.1,
                                                      random_state=3407)
    pd.DataFrame(X_train,columns=data.columns).to_csv('../data/data_splitClassifier_test/X_train{}.csv'.format(i),index=False)
    pd.DataFrame(X_test,columns=data.columns).to_csv('../data/data_splitClassifier_test/X_test{}.csv'.format(i),index=False)
    pd.DataFrame(X_val,columns=data.columns).to_csv('../data/data_splitClassifier_test/X_val{}.csv'.format(i),index=False)
    # pd.DataFrame(y_train,columns=['target']).to_csv('../data/data_splitClassifier/y_train{}.csv'.format(i),index=False)
    # pd.DataFrame(y_test,columns=['target']).to_csv('../data/data_splitClassifier/y_test{}.csv'.format(i),index=False)
    # pd.DataFrame(y_val,columns=['target']).to_csv('../data/data_splitClassifier/y_val{}.csv'.format(i),index=False)
    print('第{}折交叉验证数据保存成功'.format(i))
    i=i+1




def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def create_dataset_seq():
    df = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA(4).csv',usecols=['SMILES'])
    # print(df.columns)
    # print(df.shape)
    vocab = []
    datas = []
    for i, row in df.iterrows():
        data = row["SMILES"]
        tokens = smi_tokenizer(data).split(" ")
        if len(tokens) <= 128:
            di = tokens+["PAD"]*(128-len(tokens))
        else:
            di = tokens[:128]
        datas.append(di)
        vocab.extend(tokens)
    vocab = list(set(vocab))
    vocab = ["PAD"]+vocab
    with open("vocab.txt","w",encoding="utf8") as f:
        for i in vocab:
            f.write(i)
            f.write("\n")
    mlist = []
    word2id = {}
    for i,d in enumerate(vocab):
        word2id[d] = i
    for d_i in datas:
        mi = [word2id[d] for d in d_i]
        mlist.append(np.array(mi))
        # print(np.array(mlist).shape)
        # print(mlist)

    return mlist

seq = create_dataset_seq()
seq_numpy = np.array(seq)
seq_numpy = np.squeeze(seq_numpy)
# print(seq_numpy.shape)
# print(seq_numpy[1])
print(data.shape)
# exit()
# data = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA(4).csv')
columns_list = ['SMILES{}'.format(i+1) for i in range(seq_numpy.shape[1])]
for i in data.columns:
    if data[i].dtype=='object':
        data.drop(columns=[i],inplace=True)
print(data.shape)
data[columns_list] = seq_numpy

# y = data['label']
# data.drop(columns=['label','seq_len'],inplace=True)
print(data.shape)
# exit()
# data_scale_2 = scale(data)
data = pd.DataFrame(data,columns=data.columns)
X_stand = data.values
print(X_stand.shape)
i = 1
for train_index, test_index in kf.split(X_stand):
    X_train, X_test = X_stand[train_index], X_stand[test_index]
    # y_train, y_test = y[train_index], y[test_index]
    X_train, X_val, y_train, y_val = train_test_split(X_train, X_train,
                                                      test_size=0.1,
                                                      random_state=3407)

    pd.DataFrame(X_train,columns=data.columns).to_csv('../data/data_splitClassifier_235_test/X_train{}.csv'.format(i),index=False)
    pd.DataFrame(X_test,columns=data.columns).to_csv('../data/data_splitClassifier_235_test/X_test{}.csv'.format(i),index=False)
    pd.DataFrame(X_val,columns=data.columns).to_csv('../data/data_splitClassifier_235_test/X_val{}.csv'.format(i),index=False)
    # pd.DataFrame(y_train,columns=['label']).to_csv('../data/data_splitClassifier_235_test/y_train{}.csv'.format(i),index=False)
    # pd.DataFrame(y_test,columns=['label']).to_csv('../data/data_splitClassifier_235_test/y_test{}.csv'.format(i),index=False)
    # pd.DataFrame(y_val,columns=['label']).to_csv('../data/data_splitClassifier_235_test/y_val{}.csv'.format(i),index=False)
    print('第{}折交叉验证数据保存成功'.format(i))
    i=i+1