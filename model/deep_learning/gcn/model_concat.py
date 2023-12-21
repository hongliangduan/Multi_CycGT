import os

import torch.nn as nn
from models import Model_TGCN
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

def main():
    for num in range(10, 11):
        PATH_x_train = '../../../data/data_splitClassifier/X_train{}.csv'.format(num)
        PATH_x_val = '../../../data/data_splitClassifier/X_val{}.csv'.format(num)
        PATH_x_test = '../../../data/data_splitClassifier/X_test{}.csv'.format(num)

        df_seq_train ,y_train_tensor, y_true_train, list_num_train = func(PATH_x_train)
        df_seq_test, y_test_tensor, y_true_test, list_num_test = func(PATH_x_test)
        df_seq_val, y_val_tensor, y_true_val, list_num_val = func(PATH_x_val)

        device = 'cpu'
        node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
        edge_featurizer = CanonicalBondFeaturizer(bond_data_field='e')
        atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')
        bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat')
        mol = Chem.MolFromSmiles('c1ccccc1')
        n_feats = atom_featurizer.feat_size('feat')

        def get_data(df):
            mols = [Chem.MolFromSmiles(x) for x in df['SMILES']]
            g = [mol_to_complete_graph(m, node_featurizer=node_featurizer) for m in mols]
            y = np.array(list((df['label'])))
            y = np.array(y, dtype=np.int64)
            return g, y

        ncls = 2

        gcn_net = GCNPredictor(in_feats=n_feats,
                               hidden_feats=[60, 20],
                               n_tasks=2,
                               predictor_hidden_feats=10,
                               predictor_dropout=0.5, )

        gcn_net = gcn_net.to(device)
        model_tgcn = Model_TGCN().to(device)

        def collate(sample):
            _, list_num, graphs, labels = map(list, zip(*sample))
            batched_graph = dgl.batch(graphs)
            batched_graph.set_n_initializer(dgl.init.zero_initializer)
            batched_graph.set_e_initializer(dgl.init.zero_initializer)
            return _, list_num, batched_graph, torch.tensor(labels)

        train_X = pd.read_csv(PATH_x_train)
        x_train, y_train = get_data(train_X)
        train_data = list(zip(df_seq_train, list_num_train, x_train, y_train))
        train_loader_ = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate, drop_last=True)

        test_X = pd.read_csv(PATH_x_test)
        x_test, y_test = get_data(test_X[:688])
        test_data = list(zip(df_seq_test, list_num_test, x_test, y_test))
        test_loader_test = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=collate, drop_last=True)

        val_X = pd.read_csv(PATH_x_val)
        x_val, y_val = get_data(val_X[:624])
        val_data = list(zip(df_seq_val, list_num_val, x_val, y_val))
        val_loader_val = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=collate, drop_last=True)
        optimizer = torch.optim.Adam([{'params': gcn_net.parameters()},

                                      {'params': model_tgcn.parameters()}], lr=0.001)

        for epoch in range(1, 201):
            # training
            gcn_net.train()

            model_tgcn.train()
            train_epoch_loss, train_epoch_acc, train_epoch_r2 = 0, 0, 0
            for i, (X, list_num, graph, labels) in enumerate(train_loader_):
                train_labels = labels.to(device)
                atom_feats = graph.ndata.pop('h').to(device)
                atom_feats, train_labels = atom_feats.to(device), train_labels.to(device)
                train_pred = gcn_net(graph, atom_feats, model_use='a')


                y = model_tgcn(train_pred).to("cpu")

                y = torch.reshape(y, [16])

                train_loss = nn.BCELoss()(y, train_labels.float())
                optimizer.zero_grad()
                train_loss.requires_grad_(True)
                train_loss.backward()
                optimizer.step()
                train_epoch_loss += train_loss.detach().item()
                train_pred_cls = train_pred.argmax(-1).detach().to('cpu').numpy()
                train_true_label = train_labels.to('cpu').numpy()
                yy = [1 if i >= 0.5 else 0 for i in y.detach().numpy()]
                train_epoch_acc += sum(train_true_label == yy)
            train_epoch_acc = train_epoch_acc / train_true_label.shape[0]
            train_epoch_acc /= (i + 1)
            train_epoch_loss /= (i + 1)

            os.makedirs(f'./model/gcn/{num}/gcn/', exist_ok=True)
            os.makedirs(f'./model/gcn/{num}/g_fc/', exist_ok=True)

            torch.save(gcn_net, './model/gcn/{}/gcn/{}_gcn.pt'.format(num, epoch))
            torch.save(model_tgcn, './model/gcn/{}/g_fc/{}_g_fc.pt'.format(num, epoch))

            def train_test_val(dataloader):
                epoch_loss, epoch_acc = 0, 0
                mlist = []
                gcn_net.eval()
                model_tgcn.eval()
                for i, (X, list_num, graph, labels) in enumerate(dataloader):
                    labels = labels.to(device)
                    atom_feats = graph.ndata.pop('h').to(device)
                    atom_feats, labels = atom_feats.to(device), labels.to(device)
                    pred = gcn_net(graph, atom_feats, model_use='a')
                    y = model_tgcn(pred).to("cpu")
                    y = torch.reshape(y, [16])
                    loss = nn.BCELoss()(y, labels.float())
                    epoch_loss += loss.detach().item()
                    true_label = labels.to('cpu').numpy()
                    yy = [1 if m >= 0.5 else 0 for m in y.detach().numpy()]
                    y = y.detach().numpy()
                    mlist.extend(y)
                    epoch_acc += sum(true_label == yy)
                epoch_acc = epoch_acc / true_label.shape[0]
                epoch_acc /= (i + 1)
                epoch_loss /= (i + 1)

                return epoch_acc, epoch_loss, mlist

            test_epoch_acc, test_epoch_loss, test_list = train_test_val(test_loader_test)

            val_epoch_acc, val_epoch_loss, val_list = train_test_val(val_loader_val)

            print(f"epoch: {epoch}, train_LOSS: {train_epoch_loss:.3f}, train_ACC: {train_epoch_acc:.3f}")
            print(f"epoch: {epoch}, test_LOSS : {test_epoch_loss:.3f}, test_ACC : {test_epoch_acc:.3f}")
            print(f"epoch: {epoch}, val_LOSS  : {val_epoch_loss:.3f}, val_ACC  : {val_epoch_acc:.3f}")
            print()

            y_true_test = pd.read_csv(PATH_x_test, usecols=['label']).values
            y_true_test = y_true_test[:688]
            t1, t2 = pd.DataFrame(test_list, columns=['predict']), pd.DataFrame(y_true_test, columns=['true'])
            tt = pd.concat([t1, t2], axis=1)
            os.makedirs(f'./pred_data/gcn/{num}/test/', exist_ok=True)
            pd.DataFrame(tt).to_csv('./pred_data/gcn/{}/test/experiment_{}_predicted_test_values.csv'.format(num, epoch),
                                    index=False)

            y_true_val = pd.read_csv(PATH_x_val, usecols=['label']).values
            y_true_val = y_true_val[:624]
            t1, t2 = pd.DataFrame(val_list, columns=['predict']), pd.DataFrame(y_true_val, columns=['true'])
            tt = pd.concat([t1, t2], axis=1)
            os.makedirs(f'./pred_data/gcn/{num}/val/', exist_ok=True)
            pd.DataFrame(tt).to_csv('./pred_data/gcn/{}/val/experiment_{}_predicted_valid_values.csv'.format(num, epoch),
                                    index=False)


if __name__ == '__main__':
    main()

