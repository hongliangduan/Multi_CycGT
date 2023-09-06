import os

import torch
import numpy as np
import torch.nn as nn
from lstm_model import LSTM
import pandas as pd
# from read import read_pos
from lstm_model import create_dataset_seq,create_dataset_list,create_dataset_number,d_loadar
from sklearn.metrics import accuracy_score


# 定义超参数
input_size = 128
hidden_size = 32
num_layers = 2
output_size = 1
learning_rate = 0.0001
num_epochs = 30
batch_size = 4096

def func(x,y):
    # 位置信息
    # pos_arr, mlist = read_pos(x)
    # mlist = mlist.reshape(mlist.shape[0], mlist.shape[2])
    # 纯数字
    df_num, y_true = create_dataset_number(x, y)
    # 列表
    list_num = create_dataset_list(x)
    # 纯数字转换向量
    list_num = torch.tensor(df_num, dtype=torch.float32)

    # list_num = torch.cat([list_num, list_num], dim=1)
    # list_num = np.concatenate([list_num, mlist], axis=1)
    # list_num = torch.tensor([item for item in list_num]).cuda().to(torch.float)
    # 序列
    df_seq = create_dataset_seq(x)
    df_seq = torch.tensor([item for item in df_seq]).to(torch.int64)
    # 标签
    y = torch.tensor([item for item in y_true]).to(torch.float)

    return df_seq,list_num,y,y_true

if __name__ == '__main__':
    for i in range(1,11):
        PATH_x_train = '../../data/data_splitClassifier/X_train{}.csv'.format(i)
        # PATH_y_train = '../../data/data_splitClassifier/y_train{}.csv'.format(i)
        PATH_x_test = '../../data/data_splitClassifier/X_test{}.csv'.format(i)
        # PATH_y_test = '../../data/data_splitClassifier/y_test{}.csv'.format(i)
        PATH_x_val = '../../data/data_splitClassifier/X_val{}.csv'.format(i)
        # PATH_y_val = '../../data/data_splitClassifier/y_val{}.csv'.format(i)

        df_seq_train,list_num_train,y_train,y_true_train = func(PATH_x_train,PATH_x_train)
        df_seq_test,list_num_test,y_test,y_true_test = func(PATH_x_test,PATH_x_test)
        df_seq_val,list_num_val,y_val,y_true_val = func(PATH_x_val,PATH_x_val)


        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        # 训练模型
        for epoch in range(num_epochs):
            model.train()
            loss_avg = 0
            step = 0
            train_epoch_acc = 0
            for X, x2, y1 in d_loadar(df_seq_train,list_num_train,y_train):
                step += 1
                X = X.to(device)
                x2 = x2.to(device)
                y1 = y1.to(device)
                optimizer.zero_grad()
                x2 = x2.to(torch.float32)
                y_pred = model(X, x2)
                loss = criterion(y_pred, y1)
                loss.backward()
                optimizer.step()
                loss_avg += loss.item()
                train_true_label = y1.to('cpu').numpy()
                train_true_label = np.ravel(train_true_label)
                yy = [1 if i >= 0.5 else 0 for i in y_pred.detach().to('cpu').numpy()]
                train_epoch_acc += sum(train_true_label == yy)
            # print(train_epoch_acc)
            # print(train_true_label.shape[0])
            train_epoch_acc = train_epoch_acc / train_true_label.shape[0]
            # print(train_epoch_acc)
            train_epoch_acc /= step
            os.makedirs(f'./model/lstm_fc/{i}/', exist_ok=True)
            torch.save(model, './model/lstm_fc/{}/第{}个lstm_model.pt'.format(i,epoch))




            def train_test_val(seq,num,y):
                total = 0
                corrent = 0
                model.eval()
                with torch.no_grad():
                    test_loss = 0
                    for X, x2, y in d_loadar(seq,num,y):
                        X = X.to(device)
                        x2 = x2.to(device)
                        y = y.to(device)
                        outputs = model(X, x2)
                        y_pred = torch.round(outputs)
                        total += y.size(0)
                        corrent += (y_pred == y.float()).sum().item()

                pred = model(seq.to(device), num.to(device))
                acc = 100 * corrent / total
                return acc,pred


            acc_val,pred_val = train_test_val(df_seq_val,list_num_val,y_val)
            acc_test,pred_test = train_test_val(df_seq_test,list_num_test,y_test)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_avg / step:.4f},train_ACC: {train_epoch_acc:.4f},test_Acc:{acc_test}')

            t1,t2 = pd.DataFrame(pred_test.cpu().detach().numpy(),columns=['predict']),pd.DataFrame(y_true_test,columns=['ture'])
            tt = pd.concat([t1,t2],axis=1)
            os.makedirs(f'./pred_data/{i}/', exist_ok=True)
            pd.DataFrame(tt).to_csv('./pred_data/{}/第{}次测试集预测结果.csv'.format(i,epoch),index=False)
            threshold = 0.5
            y_pred_label = [1 if y >= threshold else 0 for y in pred_test]
            # print(y_pred_label)
            print('acc:',accuracy_score(y_pred_label,y_true_test))
            # print('第{}次运行结束\n'.format(i))