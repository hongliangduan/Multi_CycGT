import os
import pickle
import torch.nn as nn
from models import Transformer_test, Model_TGCN, batch_size
from data_pretreatment import func
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import DataLoader
import dgl
from dgllife.utils import *
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor
torch.cuda.empty_cache()

def main_():
    for num in range(1, 11):
        PATH_x_train = '../../../data/data_splitClassifier/X_train{}.csv'.format(num)
        PATH_x_test = '../../../data/data_splitClassifier/X_test{}.csv'.format(num)
        PATH_x_val = '../../../data/data_splitClassifier/X_val{}.csv'.format(num)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_trans = Transformer_test().to(device)
        torch.cuda.empty_cache()
        df_seq_train, y_train_tensor, y_true_train, list_num_train = func(PATH_x_train)
        df_seq_test, y_test_tensor, y_true_test, list_num_test = func(PATH_x_test)
        df_seq_val, y_val_tensor, y_true_val, list_num_val = func(PATH_x_val)

        # 数据转换图格式
        device = 'cpu'


        def get_data(df):

            y = np.array(list((df['label'])))
            y = np.array(y, dtype=np.int64)
            return  y

        model_tgcn = Model_TGCN().to(device)

        def collate(sample):
            _, labels, index = map(list, zip(*sample))
            return _, torch.tensor(labels), index

        train_X = pd.read_csv(PATH_x_train)
        y_train = get_data(train_X)
        train_data = list(zip(df_seq_train, y_train, [i for i in range(len(train_X))]))
        train_loader_ = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)

        test_X = pd.read_csv(PATH_x_test)
        y_test = get_data(test_X)
        test_data = list(zip(df_seq_test, y_test, [i for i in range(len(test_X))]))
        test_loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate,
                                      drop_last=True)

        val_X = pd.read_csv(PATH_x_val)
        y_val = get_data(val_X)
        val_data = list(zip(df_seq_val, y_val, [i for i in range(len(val_X))]))
        val_loader_val = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True)


        optimizer = torch.optim.Adam([{'params': model_trans.parameters()},
                                      {'params': model_tgcn.parameters()}], lr=0.001)

        for epoch in range(1, 501):
            # train
            model_trans.train()
            model_tgcn.train()
            train_epoch_loss, train_epoch_acc, train_epoch_r2 = 0, 0, 0
            for i, (X,labels, index) in enumerate(train_loader_):
                train_labels = labels.to(device)
                X = torch.cat(X, dim=0)
                X = torch.reshape(X, [batch_size, 128])
                X = X.to("cuda")
                y = model_trans(X)
                y = model_tgcn(y).to("cpu")
                y = torch.reshape(y, [batch_size])
                train_loss = nn.BCELoss()(y, train_labels.float())
                optimizer.zero_grad()
                train_loss.requires_grad_(True)
                train_loss.backward()
                optimizer.step()
                train_epoch_loss += train_loss.detach().item()
                train_true_label = train_labels.to('cpu').numpy()
                yy = [1 if i >= 0.5 else 0 for i in y.detach().numpy()]
                train_epoch_acc += sum(train_true_label == yy)

            train_epoch_acc = train_epoch_acc / train_true_label.shape[0]
            train_epoch_acc /= (i + 1)
            train_epoch_loss /= (i + 1)


            os.makedirs(f'./model_origin/gcn_transformer_fc/{num}/transformer/', exist_ok=True)
            os.makedirs(f'./model_origin/gcn_transformer_fc/{num}/tgcn/', exist_ok=True)

            torch.save(model_trans,
                       './model_origin/gcn_transformer_fc/{}/transformer/{}_transformer.pt'.format(num, epoch))
            torch.save(model_tgcn, './model_origin/gcn_transformer_fc/{}/tgcn/{}_tgcn.pt'.format(num, epoch))

            def train_test_val(dataloader):
                epoch_loss, epoch_acc = 0, 0
                mlist = []
                model_trans.eval()
                model_tgcn.eval()
                for i, (X, labels, index) in enumerate(dataloader):
                    labels = labels.to(device)
                    # Using GCN to obtain GCN encoding for sequence information
                    X = torch.cat(X, dim=0)
                    X = torch.reshape(X, [batch_size, 128])
                    X = X.to("cuda")
                    y = model_trans(X)
                    y = model_tgcn(y).to("cpu")
                    y = torch.reshape(y, [batch_size])
                    loss = nn.BCELoss()(y, labels.float())
                    epoch_loss += loss.detach().item()
                    pred_cls = y.detach().numpy()
                    true_label = labels.to('cpu').numpy()
                    yy = [1 if m >= 0.5 else 0 for m in y.detach().numpy()]
                    mlist.extend(pred_cls)
                    epoch_acc += sum(true_label == yy)
                epoch_acc = epoch_acc / true_label.shape[0]
                epoch_acc /= (i + 1)
                epoch_loss /= (i + 1)

                return epoch_acc, epoch_loss, mlist

            test_epoch_acc, test_epoch_loss, test_list = train_test_val(test_loader_test)
            val_epoch_acc, val_epoch_loss, val_list = train_test_val(val_loader_val)

            print(f"epoch: {epoch}, train_LOSS      : {train_epoch_loss:.3f}, train_ACC        : {train_epoch_acc:.3f}")
            print(f"epoch: {epoch}, test_LOSS       : {test_epoch_loss:.3f}, test_ACC         : {test_epoch_acc:.3f}")
            print(f"epoch: {epoch}, val_LOSS        : {val_epoch_loss:.3f}, val_ACC          : {val_epoch_acc:.3f}")

            y_true_test = pd.read_csv(PATH_x_test, usecols=['label']).values
            t1, t2 = pd.DataFrame(test_list, columns=['predict']), pd.DataFrame(y_true_test, columns=['true'])
            tt = pd.concat([t1, t2], axis=1)
            os.makedirs(f'./pred_data_origin/gcn_transformer_fc/{num}/test/', exist_ok=True)
            pd.DataFrame(tt).to_csv('./pred_data_origin/gcn_transformer_fc/{}/test/experiment_{}_predicted_test_values.csv'.format(num, epoch),index=False)
            y_true_val = pd.read_csv(PATH_x_val, usecols=['label']).values
            t1, t2 = pd.DataFrame(val_list, columns=['predict']), pd.DataFrame(y_true_val, columns=['true'])
            tt = pd.concat([t1, t2], axis=1)
            os.makedirs(f'./pred_data_origin/gcn_transformer_fc/{num}/val/', exist_ok=True)
            pd.DataFrame(tt).to_csv('./pred_data_origin/gcn_transformer_fc/{}/val/experiment_{}_predicted_valid_values.csv'.format(num, epoch),index=False)




if __name__ == '__main__':
    main_()

