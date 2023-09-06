#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('./CycPeptMPDB_Peptide_Assay_PAMPA(4).csv')



data.head()



for i in range(len(data)):
    data['Sequence'][i] = eval(data['Sequence'][i])
object_lost = list(set(data['Sequence'].sum()))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(object_lost)



import numpy as np


# Sequence_list = ['Sequence{}'.format(i+1) for i in range(15)]
# Sequence = pd.DataFrame(columns=Sequence_list,index=range(len(data)))





# for i in range(len(data)):
#     Sequence_transorform = le.transform(data['Sequence'][i])
#     if Sequence_transorform.shape[0]<15:
#         Sequence.loc[i] = np.hstack([Sequence_transorform,np.zeros(((15-Sequence_transorform.shape[0])),)])
#     else:
#         Sequence.loc[i] = Sequence_transorform
#
#
# data[Sequence_list] = Sequence.values


from rdkit import Chem
import numpy as np
def read_mol_file(filename):
    """
    读取mol文件中原子的坐标
    :param filename: mol文件名
    :return: 原子坐标列表
    """
    mol = Chem.MolFromMolFile(filename, sanitize=False)
    atoms = mol.GetAtoms()
    atom_coords = []
    for atom in atoms:
        atom_coord = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        atom_coords.append([atom_coord.x, atom_coord.y, atom_coord.z])
    return atom_coords
idarr = np.zeros((data.shape[0],3))
for i in range(data['CycPeptMPDB_ID'].shape[0]):
    id_num = data['CycPeptMPDB_ID'][i]
    iddata = np.array(read_mol_file('../cycy/id_{}.mol'.format(id_num)))
    ones_arr = np.ones((1,iddata.shape[0]))
    idarr[i,:] = np.dot(ones_arr,iddata)
#     if iddata.shape[0]!=127:
#         idarr[i,:,:] = np.vstack([iddata,np.zeros((127-len(iddata),3))])
#     else:
#         idarr[i,:,:] = iddata




final_ones = np.ones((3,128))


# In[11]:


res_arr = np.dot(idarr,final_ones)


# In[12]:


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


# In[13]:


# #可以使用一维卷积操作来提取特征并压缩维度。以下是一种可能的方法：
# #。在训练过程中，Keras会自动提取特征并压缩维度。最后，我们使用训练好的模型来预测数据并压缩维度。输出应该是（100,1）的形状。
# import numpy as np
# from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
# from keras.models import Sequential
# import tensorflow as tf
# from tensorflow.keras import layers
# # 创建一个包含一维卷积层的模型
# model = Sequential()
# model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=( 127,3)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))

# # 编译模型
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # 训练模型
# model.fit(idarr, y, epochs=100, batch_size=32)

# # 使用模型预测并压缩数据维度
# compressed_data = model.predict(idarr)
# print(compressed_data.shape)




CycPeptMPDB_ID_list = ['CycPeptMPDB_ID{}'.format(i) for i in range(128)]



df_X[CycPeptMPDB_ID_list] = res_arr

print(df_X.shape)




from sklearn.model_selection import KFold,train_test_split
# 创建10折交叉验证的对象
kf = KFold(n_splits=10, shuffle=True, random_state=42)
i=1
# 循环遍历10次,训练和测试模型
X_stand = df_X.values
for train_index, test_index in kf.split(X_stand):
    X_train, X_test = X_stand[train_index], X_stand[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.1,
                                                      random_state=42)
    pd.DataFrame(X_train,columns=df_X.columns).to_csv('../data/data_splitClassifier_362/X_train{}.csv'.format(i),index=False)
    pd.DataFrame(X_test,columns=df_X.columns).to_csv('../data/data_splitClassifier_362/X_test{}.csv'.format(i),index=False)
    pd.DataFrame(X_val,columns=df_X.columns).to_csv('../data/data_splitClassifier_362/X_val{}.csv'.format(i),index=False)
    pd.DataFrame(y_train,columns=['target']).to_csv('../data/data_splitClassifier_362/y_train{}.csv'.format(i),index=False)
    pd.DataFrame(y_test,columns=['target']).to_csv('../data/data_splitClassifier_362/y_test{}.csv'.format(i),index=False)
    pd.DataFrame(y_val,columns=['target']).to_csv('../data/data_splitClassifier_362/y_val{}.csv'.format(i),index=False)
    print('第{}折交叉验证数据保存成功'.format(i))
    i=i+1






