import os
import torch
import torch.nn as nn
from cnn_process import CNN
import pandas as pd
from cnn_process import create_dataset_seq, create_dataset_list, create_dataset_number, d_loadar
import warnings

warnings.filterwarnings('ignore')
torch.manual_seed(3407)
# Define hyperparameter
batch_size = 64
learning_rate = 0.0001
epochs = 30
input_size = 128
hidden_size = 32
num_layers = 2
output_size = 103


def func(x, y):
    df_num, y_true = create_dataset_number(x, y)
    df_seq = create_dataset_seq(x)
    df_seq = torch.tensor([item for item in df_seq]).to(torch.int64)
    y = torch.tensor([item for item in y_true]).to(torch.float)
    return df_seq, y, y_true

if __name__ == '__main__':
    for i in range(1, 11):
        PATH_x_train = '../../../data/data_splitClassifier/X_train{}.csv'.format(i)
        PATH_x_val = '../../../data/data_splitClassifier/X_val{}.csv'.format(i)
        PATH_x_test = '../../../data/data_splitClassifier/X_test{}.csv'.format(i)

        df_seq_train, y_train, y_true_train = func(PATH_x_train, PATH_x_train)
        df_seq_val, y_val, y_true_val = func(PATH_x_val, PATH_x_val)
        df_seq_test, y_test, y_true_test = func(PATH_x_test, PATH_x_test)

        model = CNN(input_size, hidden_size, num_layers, output_size)
        criterion = nn.BCELoss()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            loss_avg = 0
            step = 0
            total = 0
            corrent = 0
            for X, y1 in d_loadar(df_seq_train, y_train):
                step += 1

                optimizer.zero_grad()
                X = X.to(torch.float32)
                y_pred = model(X)
                loss = criterion(y_pred, y1)
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                loss_avg += loss.item()
                y_pred = torch.round(y_pred)
                total += y1.size(0)
                corrent += (y_pred == y1.float()).sum().item()
            print(corrent)
            print(total)
            train_acc = corrent / total
            print(train_acc)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_avg / step:.4f}')

            def train_test_val(seq,y):
                total = 0
                corrent = 0
                model.eval()
                with torch.no_grad():
                    for X, y1 in d_loadar(seq, y):
                        X = X.to(torch.float32)
                        outputs = model(X)
                        y_pred = torch.round(outputs)
                        total += y1.size(0)
                        corrent += (y_pred == y1.float()).sum().item()
                seq = seq.to(torch.float32).cpu()
                pred = model(seq.cpu())
                acc = 100 * corrent / total

                return acc, pred

            os.makedirs(f'./model/cnn_fc/{i}/', exist_ok=True)

            torch.save(model, './model/cnn_fc/{}/{}_cnn_fc_model.pt'.format(i, epoch))
            val_acc, val_pred = train_test_val(df_seq_val, y_val, )
            test_acc, test_pred = train_test_val(df_seq_test, y_test, )
            print('train_acc:', train_acc, 'test_acc：', test_acc)

            t1, t2 = pd.DataFrame(test_pred.cpu().detach().numpy(), columns=['predict']), \
                pd.DataFrame(y_true_test, columns=['true'])
            tt = pd.concat([t1, t2], axis=1)

            os.makedirs(f'./pred_data/cnn/{i}/', exist_ok=True)

            pd.DataFrame(tt).to_csv('./pred_data/cnn/{}/experiment_{}_predicted_values.csv'.format(i, epoch), index=False)
