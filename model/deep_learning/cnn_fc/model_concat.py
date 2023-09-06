import os

import torch
import numpy as np
import torch.nn as nn
from cnn_process import CNN
import pandas as pd
# from read import read_pos
from cnn_process import create_dataset_seq ,create_dataset_list ,create_dataset_number ,d_loadar
from sklearn.metrics import f1_score ,precision_score ,recall_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(3407)
# 定义超参数
batch_size = 64
learning_rate = 0.0001
epochs = 30
input_size = 128
hidden_size = 32
num_layers = 2
output_size = 103

# 对 化学性质 列表 坐标位置 smiles码 进行 数据处理
def func(x ,y):


    # 对 化学性质  进行 数据处理
    df_num, y_true = create_dataset_number(x, y)
    # 对 列表  进行 数据处理
    # list_num = create_dataset_list(x)
    # 对 smiles码 进行 数据处理
    df_seq = create_dataset_seq(x)
    # 化学性质
    list_num = torch.tensor(df_num, dtype=torch.float32)
    # 化学性质 + 列表
    # list_num = torch.cat([list_num, tensor_data_num], dim=1)


    # 化学性质 + 列表  转换向量
    # list_num = torch.tensor(list_num).cuda().to(torch.float)
    # smies码 转换向量
    df_seq = torch.tensor([item for item in df_seq]).to(torch.int64)

    # labels 转换向量
    y = torch.tensor([item for item in y_true]).to(torch.float)

    return df_seq ,list_num ,y ,y_true

if __name__ == '__main__':
    
    for i in range(1, 11):
        # 读取数据
        PATH_x_train = '../../data/data_splitClassifier/X_train{}.csv'.format(i)
        # PATH_y_train = '../../data/data_splitClassifier/y_train{}.csv'.format(i)
        PATH_x_val = '../../data/data_splitClassifier/X_val{}.csv'.format(i)
        # PATH_y_val = '../../data/data_splitClassifier/y_val{}.csv'.format(i)
        PATH_x_test = '../../data/data_splitClassifier/X_test{}.csv'.format(i)
        # PATH_y_test = '../../data/data_splitClassifier/y_test{}.csv'.format(i)


        # 将训练集和测试集封装为 Dataset 对象
        # 将 Dataset 对象封装为 DataLoader 对象，用于批次训练
        df_seq_train,list_num_train, y_train, y_true_train = func(PATH_x_train, PATH_x_train)
        df_seq_val, list_num_val, y_val, y_true_val = func(PATH_x_val, PATH_x_val)
        df_seq_test, list_num_test, y_test, y_true_test = func(PATH_x_test, PATH_x_test)


    
        # 初始化模型并设定损失函数和优化器
        model = CNN(input_size, hidden_size, num_layers, output_size)
        criterion = nn.BCELoss()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 训练和验证
        for epoch in range(epochs):
            model.train()
            loss_avg = 0
            step = 0
            total = 0
            corrent = 0
            for X, x2, y1 in d_loadar(df_seq_train, list_num_train, y_train):
                step += 1

                optimizer.zero_grad()
                x2 = x2.to(torch.float32)
                X = X.to(torch.float32)

                y_pred = model(X, x2)

                loss = criterion(y_pred, y1)
                loss.backward()
                optimizer.step()
                loss_avg += loss.item()
                # torch.save(model.state_dict(), './model/cnn_fc/第{}个cnn_fc_model.pt'.format(i))
                y_val_pred = torch.round(y_pred)
                total += y1.size(0)
                corrent += (y_val_pred == y1.float()).sum().item()
            train_acc = 100 * corrent / total
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_avg / step:.4f}')
            print('train_acc:', train_acc)

            def train_test_val(seq,num,y):
                total = 0
                corrent = 0
                model.eval()
                with torch.no_grad():
                    for X, x2, y in d_loadar(seq, num, y):

                        x2 = x2.to(torch.float32)
                        X = X.to(torch.float32)
                        outputs = model(X, x2)
                        y_pred = torch.round(outputs)
                        total += y.size(0)
                        corrent += (y_pred == y.float()).sum().item()
                seq = seq.to(torch.float32).cpu()
                num = num.to(torch.float32).cpu()
                pred = model(seq.cpu(), num.cpu())
                acc = 100 * corrent / total

                return acc,pred
            os.makedirs(f'./model/cnn_fc/{i}/', exist_ok=True)
            # 保存模型
            torch.save(model,'./model/cnn_fc/{}/第{}个cnn_fc_model.pt'.format(i,epoch))
            val_acc,val_pred = train_test_val(df_seq_val,list_num_val,  y_val,)
            test_acc,test_pred = train_test_val( df_seq_test,list_num_test, y_test,)
            print("val_acc",val_acc)
            print("test_acc",test_acc)

            t1, t2 = pd.DataFrame(test_pred.cpu().detach().numpy(), columns=['predict']),\
                        pd.DataFrame(y_true_test,columns=['true'])
            tt = pd.concat([t1, t2], axis=1)

            os.makedirs(f'./pred_data/cnn_fc/{i}/', exist_ok=True)
            pd.DataFrame(tt).to_csv('./pred_data/cnn_fc/{}/第{}次测试集预测值.csv'.format(i,epoch), index=False)

