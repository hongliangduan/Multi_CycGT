
import torch
import numpy as np
import torch.nn as nn
from transformer_model import LSTM,BERTTextClassificationModel
from transformer_torch import Transformer
import pandas as pd
from transformer_model import create_dataset_seq,create_dataset_list,create_dataset_number,d_loadar
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
# 定义超参数
input_size = 128
hidden_size = 32
num_layers = 2
output_size = 1
learning_rate = 0.01
num_epochs = 50
batch_size = 2
acc_list = []
test_loss_list = []
f1_list = []
precision_list = []
recall_list = []


# 对 化学性质 列表 坐标位置 smiles码 进行 数据处理
def func(PATH):

    # 对 化学性质  进行 数据处理
    df_num, y_true = create_dataset_number(PATH)
    # 对 列表  进行 数据处理
    list_num = create_dataset_list(PATH)
    # 对 smiles码 进行 数据处理
    df_seq = create_dataset_seq(PATH)
    # smies码 转换向量
    df_seq = torch.tensor([item for item in df_seq]).to(torch.int64)

    # 化学性质
    tensor_data_num = torch.tensor(df_num, dtype=torch.float32)
    # 化学性质 + 列表
    list_num = torch.cat([list_num, tensor_data_num], dim=1)
    # labels 转换向量
    y = torch.tensor([item for item in y_true]).to(torch.float)

    return df_seq, y, y_true

if __name__ == '__main__':

    for i in range(1,11):
        # 读取数据
        PATH_x_train = '../../data/data_splitClassifier/X_train{}.csv'.format(i)
        PATH_x_val = '../../data/data_splitClassifier/X_val{}.csv'.format(i)
        PATH_x_test = '../../data/data_splitClassifier/X_test{}.csv'.format(i)

        
        # 初始化模型并设定损失函数和优化器
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
        model = Transformer().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        torch.cuda.empty_cache()
        # 将训练集和测试集封装为 Dataset 对象
        df_seq_train,y_train,y_true_train = func(PATH_x_train)
        print(df_seq_train)
        df_seq_val, y_val, y_true_val = func(PATH_x_val,)
        df_seq_test,y_test,y_true_test = func(PATH_x_test)

        train_dataset = torch.utils.data.TensorDataset(df_seq_train, y_train)
        print(train_dataset)
        val_dataset = torch.utils.data.TensorDataset(df_seq_val, y_val)
        test_dataset = torch.utils.data.TensorDataset(df_seq_test, y_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 训练和验证
        for epoch in range(num_epochs):
            model.train()
            loss_avg = 0
            step = 0
            total = 0
            corrent = 0
            for X, y in train_dataset:
                step += 1


                X = X.to(device)
                X = X.view(1,X.size(0))
                y = y.to(device)
                optimizer.zero_grad()
                y_pred = model(X)
                y_pred = y_pred.view(1)

                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                loss_avg += loss.item()
                y_val_pred = torch.round(y_pred)
                total += y.size(0)
                corrent += (y_val_pred == y.float()).sum().item()
            train_acc = 100 * corrent / total
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_avg / step:.4f}')
            print('train_acc:', train_acc)
            torch.save(model.state_dict(), './model/transformer_fc/第{}个transformer_model.pt'.format(i))



            # model.load_state_dict(torch.load('./model/transformer_fc/第{}个transformer_model.pt'.format(i)))
            #
            # model.eval()
            # total = 0
            # corrent = 0
            # with torch.no_grad():
            #     test_loss = 0
            #     for X, x2, y in d_loadar(df_seq_val, list_num_val, y_val):
            #         X = X.to(device)
            #         x2 = x2.to(device)
            #         y = y.to(device)
            #         outputs = model(X, x2)
            #         y_val_pred = torch.round(outputs)
            #         total += y.size(0)
            #         corrent += (y_val_pred == y.float()).sum().item()
            #
            # pred_val = model(df_seq_val.to(device), list_num_val.to(device))
            # val_acc = 100 * corrent / total
            # print(val_acc)

        # torch.save(model.state_dict(), './model/lstm_fc/第{}个lstm_model.pt'.format(i))

        total = 0
        corrent = 0
        # # model.load_state_dict(torch.load('./model/lstm_fc/第{}个lstm_model.pt'.format(i)))
        # with torch.no_grad():
        #     test_loss = 0
        #     for X, x2, y in d_loadar(df_seq_test, list_num_test, y_test):
        #         X = X.to(device)
        #         x2 = x2.to(device)
        #         y = y.to(device)
        #         outputs = model(X, x2)
        #         y_test_pred = torch.round(outputs)
        #         total += y.size(0)
        #         corrent += (y_test_pred == y.float()).sum().item()
        #
        # pred_test = model(df_seq_test.to(device), list_num_test.to(device))
        # # print(f"预测值：{pred_test}")
        # t1,t2 = pd.DataFrame(pred_test.cpu().detach().numpy(),columns=['predict']),pd.DataFrame(y_true_test,columns=['ture'])
        # tt = pd.concat([t1,t2],axis=1)
        # pd.DataFrame(tt).to_csv('./pred_data/transformer_fc/第{}次测试集预测值.csv'.format(i),index=False)
        # test_acc = 100 * corrent / total
        # threshold = 0.5
        # y_pred_label = [1 if y >= threshold else 0 for y in pred_test]
        # # print(y_pred_label)
        # print('acc:',accuracy_score(y_pred_label,y_true_test))
        # f1_list.append(f1_score(y_pred_label, y_true_test))
        # precision_list.append(precision_score(y_pred_label, y_true_test))
        # recall_list.append(recall_score(y_pred_label, y_true_test))
        # acc_list.append(accuracy_score(y_pred_label, y_true_test))
        # test_loss_list.append(test_loss)
        # print('第{}次运行结束\n'.format(i))
