import torch
import torch.nn as nn
from transformer_torch import Transformer
from transformer_model import create_dataset_seq ,create_dataset_list ,create_dataset_number ,d_loadar
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import DataLoader
import dgl
from dgllife.utils import *
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor


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
    # list_num = create_dataset_list(PATH)
    # 对 smiles码 进行 数据处理
    df_seq = create_dataset_seq(PATH)
    # smies码 转换向量
    df_seq = torch.tensor([item for item in df_seq]).to(torch.int64)

    # 化学性质
    tensor_data_num = torch.tensor(df_num, dtype=torch.float32)
    # 化学性质 + 列表
    # list_num = torch.cat([list_num, tensor_data_num], dim=1)
    #


    # labels 转换向量
    y = torch.tensor([item for item in y_true]).to(torch.float)

    return df_seq, y, y_true,tensor_data_num


# for i in range(1 ,2):
#     # 读取数据
#     PATH_x_train = '../../data/data_splitClassifier/X_train{}.csv'.format(i)
#     PATH_x_val = '../../data/data_splitClassifier/X_val{}.csv'.format(i)
#     PATH_x_test = '../../data/data_splitClassifier/X_test{}.csv'.format(i)
#
#
#     # 初始化模型并设定损失函数和优化器
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     # model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
#     model_trans = Transformer().to(device)
#     optimizer = torch.optim.Adam(model_trans.parameters(), lr=learning_rate)
#     criterion = nn.BCELoss()
#     torch.cuda.empty_cache()
#     # 将训练集和测试集封装为 Dataset 对象
#     df_seq_train ,y_train ,y_true_train,list_num_train = func(PATH_x_train)
#     print(df_seq_train)
#     # exit()
#
#
#     df_seq_test ,y_test ,y_true_test,list_num_test = func(PATH_x_test)
#     df_seq_val ,y_val ,y_true_val,list_num_val = func(PATH_x_val)





for num in range(1, 11):
    # 读取数据

    PATH_x_train = '../../data/data_splitClassifier/X_train{}.csv'.format(num)
    PATH_x_val = '../../data/data_splitClassifier/X_val{}.csv'.format(num)
    PATH_x_test = '../../data/data_splitClassifier/X_test{}.csv'.format(num)


    # 初始化模型并设定损失函数和优化器
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    model_trans = Transformer().to(device)
    # optimizer = torch.optim.Adam(model_trans.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    torch.cuda.empty_cache()
    # 将训练集和测试集封装为 Dataset 对象
    df_seq_train ,y_train_tensor ,y_true_train,list_num_train = func(PATH_x_train)
    print(df_seq_train)
    # exit()


    df_seq_test ,y_test_tensor ,y_true_test,list_num_test = func(PATH_x_test)
    df_seq_val ,y_val_tensor ,y_true_val,list_num_val = func(PATH_x_val)



    device = 'cpu'

    PATH_x_train = '../../data/data_splitClassifier/X_train{}.csv'.format(num)
    PATH_x_val = '../../data/data_splitClassifier/X_val{}.csv'.format(num)
    PATH_x_test = '../../data/data_splitClassifier/X_test{}.csv'.format(num)



    node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    edge_featurizer = CanonicalBondFeaturizer(bond_data_field='e')
    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat')
    mol=Chem.MolFromSmiles('c1ccccc1')
    print(node_featurizer(mol)['h'].shape)
    print(edge_featurizer(mol)['e'].shape)
    n_feats = atom_featurizer.feat_size('feat')


    def get_data(df):
        mols = [Chem.MolFromSmiles(x) for x in df['SMILES']]
        g = [mol_to_complete_graph(m, node_featurizer=node_featurizer) for m in mols]
        y = np.array(list((df['label'])))
        y = np.array(y, dtype=np.int64)
        return g,y

    ncls=2

    gcn_net = GCNPredictor(in_feats=n_feats,
                        hidden_feats=[60,20],
                        n_tasks=2,
                        predictor_hidden_feats=10,
                        predictor_dropout=0.5,)

    gcn_net = gcn_net.to(device)



    def collate(sample):
        _,list_num,graphs, labels = map(list,zip(*sample))
        batched_graph = dgl.batch(graphs)
        batched_graph.set_n_initializer(dgl.init.zero_initializer)
        batched_graph.set_e_initializer(dgl.init.zero_initializer)
        return _,list_num,batched_graph, torch.tensor(labels)


    train_X = pd.read_csv(PATH_x_train)
    x_train, y_train = get_data(train_X)
    train_data = list(zip(df_seq_train,list_num_train,x_train, y_train))
    print("len(x_train):",len(x_train))
    train_loader_ = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate, drop_last=True)

    test_X = pd.read_csv(PATH_x_test)
    x_test, y_test = get_data(test_X)
    test_data = list(zip(df_seq_test,list_num_test,x_test, y_test))
    test_loader_test = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=collate, drop_last=True)

    val_X = pd.read_csv(PATH_x_val)
    x_val, y_val = get_data(val_X)
    val_data = list(zip(df_seq_val,list_num_val,x_val, y_val))
    val_loader_val = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=collate, drop_last=True)

    optimizer = torch.optim.Adam([{'params': gcn_net.parameters()},
    {'params': model_trans.parameters()}], lr=0.001)
    # optimizer = torch.optim.Adam( gcn_net.parameters(), lr=0.001)
    def r2_loss(output, target):
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - output) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2
    best_acc = 0

    for epoch in range(1,501):

        # 训练
        gcn_net.train()
        train_epoch_loss,train_epoch_acc,train_epoch_r2 = 0,0,0
        for i, (X,list_num,graph, labels) in enumerate(train_loader_):
            train_labels = labels.to(device)
            atom_feats = graph.ndata.pop('h').to(device)
            atom_feats, train_labels = atom_feats.to(device), train_labels.to(device)

            # 序列信息走GCN 得到GCN编码
            train_pred = gcn_net(graph, atom_feats,model_use='a')
            X = torch.cat(X,dim=0)
            X = torch.reshape(X,[16,128])
            X = X.to("cuda")

            #理化性质
            list_num = torch.tensor([item.cpu().detach().numpy() for item in list_num]).cuda()
            #GCN编码+原序列信息+理化性质 传入transformer
            y = model_trans(X,train_pred,list_num).to("cpu")

            y = torch.reshape(y,[16])
            train_loss = nn.BCELoss()(y, train_labels.float())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.detach().item()
            train_pred_cls = train_pred.argmax(-1).detach().to('cpu').numpy()
            train_true_label = train_labels.to('cpu').numpy()
            yy = [1 if i >= 0.5 else 0 for i in y.detach().numpy()]
            train_epoch_acc += sum(train_true_label==yy)
        train_epoch_acc = train_epoch_acc / train_true_label.shape[0]
        train_epoch_acc /= (i + 1)
        train_epoch_loss /= (i + 1)

        torch.save(gcn_net, './model/gcn_transformer_fc/{}/gcn/第{}个gcn.pt'.format(num,epoch))
        torch.save(model_trans, './model/gcn_transformer_fc/{}/transformer/第{}个transformer.pt'.format(num,epoch))

        # gcn_net = torch.load('./model/gcn_transfrom_fc/gcn/第{}个gcn.pt'.format(epoch))
        # model_trans = torch.load('./model/gcn_transfrom_fc/transformer/第{}个transformer.pt'.format(epoch))

        # 测试
        test_epoch_loss, test_epoch_acc = 0, 0
        mlist = []
        # gcn_net.eval()
        # model_trans.eval()
        for i, (X, list_num, graph, labels) in enumerate(test_loader_test):
            # print(i)
            test_labels = labels.to(device)
            atom_feats = graph.ndata.pop('h').to(device)
            atom_feats, test_labels = atom_feats.to(device), test_labels.to(device)

            # 序列信息走GCN 得到GCN编码
            test_pred = gcn_net(graph, atom_feats,model_use='a')
            X = torch.cat(X, dim=0)
            X = torch.reshape(X, [16, 128])
            X = X.to("cuda")

            # 理化性质
            list_num = torch.tensor([item.cpu().detach().numpy() for item in list_num]).cuda()
            # GCN编码+原序列信息+理化性质 传入transformer
            y = model_trans(X, test_pred, list_num).to("cpu")

            y = torch.reshape(y, [16])
            test_loss = nn.BCELoss()(y, test_labels.float())
            # optimizer.zero_grad()
            # test_loss.backward()
            # optimizer.step()
            test_epoch_loss += test_loss.detach().item()
            test_pred_cls = test_pred.argmax(-1).detach().to('cpu').numpy()
            test_true_label = test_labels.to('cpu').numpy()
            yy = [1 if m >= 0.5 else 0 for m in y.detach().numpy()]
            mlist.extend(yy)
            # print(yy)
            # print(test_true_label)
            test_epoch_acc += sum(test_true_label == yy)

        test_epoch_acc = test_epoch_acc / test_true_label.shape[0]
        test_epoch_acc /= (i + 1)
        test_epoch_loss /= (i + 1)
        # if best_acc < train_epoch_acc:
        #     torch.save(gcn_net, './model/gcn_transfrom_fc/gcn/{}_best_gcn.pt'.format(epoch))
        #     torch.save(model_trans, './model/gcn_transfrom_fc/transformer/{}_best_transformer.pt'.format(epoch))
        #     print(f"model update:best_acc from {best_acc} to {train_epoch_acc}")
        #     best_acc = train_epoch_acc

        # print(f"epoch: {epoch + 1}, train_LOSS: {train_epoch_loss:.3f}, train_ACC: {train_epoch_acc:.3f}")
        # print(f"epoch: {epoch + 1}, test_LOSS : {test_epoch_loss:.3f}, test_ACC : {test_epoch_acc:.3f}")

        # print(f"预测值长度：{len(mlist)}")
        y_true_test = pd.read_csv(PATH_x_test,usecols=['label']).values
        y_true_test = y_true_test[:688]
        # print(f"真实值长度：{y_true_test.shape}")
        t1,t2 = pd.DataFrame(mlist,columns=['predict']),pd.DataFrame(y_true_test,columns=['true'])

        tt = pd.concat([t1,t2],axis=1)
        pd.DataFrame(tt).to_csv('./pred_data/gcn_transformer_fc/{}/test/第{}次测试集预测值.csv'.format(num,epoch),index=False)


        val_epoch_loss, val_epoch_acc = 0, 0
        mlist = []
        for i, (X, list_num, graph, labels) in enumerate(val_loader_val):
            # print(i)
            val_labels = labels.to(device)
            atom_feats = graph.ndata.pop('h').to(device)
            atom_feats, val_labels = atom_feats.to(device), val_labels.to(device)

            # 序列信息走GCN 得到GCN编码
            val_pred = gcn_net(graph, atom_feats, model_use='a')
            X = torch.cat(X, dim=0)
            X = torch.reshape(X, [16, 128])
            X = X.to("cuda")

            # 理化性质
            list_num = torch.tensor([item.cpu().detach().numpy() for item in list_num]).cuda()
            # GCN编码+原序列信息+理化性质 传入transformer
            y = model_trans(X, val_pred, list_num).to("cpu")

            y = torch.reshape(y, [16])
            val_loss = nn.BCELoss()(y, val_labels.float())
            # optimizer.zero_grad()
            # val_loss.backward()
            # optimizer.step()
            val_epoch_loss += val_loss.detach().item()
            val_pred_cls = val_pred.argmax(-1).detach().to('cpu').numpy()
            val_true_label = val_labels.to('cpu').numpy()
            yy = [1 if m >= 0.5 else 0 for m in y.detach().numpy()]
            mlist.extend(yy)
            # print(yy)
            # print(val_true_label)
            val_epoch_acc += sum(val_true_label == yy)

        val_epoch_acc = val_epoch_acc / val_true_label.shape[0]
        val_epoch_acc /= (i + 1)
        val_epoch_loss /= (i + 1)
        # if best_acc < train_epoch_acc:
        #     torch.save(gcn_net, './model/gcn_transfrom_fc/gcn/{}_best_gcn.pt'.format(epoch))
        #     torch.save(model_trans, './model/gcn_transfrom_fc/transformer/{}_best_transformer.pt'.format(epoch))
        #     print(f"model update:best_acc from {best_acc} to {train_epoch_acc}")
        #     best_acc = train_epoch_acc
        print(f"epoch: {epoch }, train_LOSS: {train_epoch_loss:.3f}, train_ACC: {train_epoch_acc:.3f}")
        print(f"epoch: {epoch }, test_LOSS : {test_epoch_loss:.3f}, test_ACC : {test_epoch_acc:.3f}")
        print(f"epoch: {epoch }, val_LOSS  : {val_epoch_loss:.3f}, val_ACC  : {val_epoch_acc:.3f}")

        # print(f"预测值长度：{len(mlist)}")
        y_true_val = pd.read_csv(PATH_x_val, usecols=['label']).values
        y_true_val = y_true_val[:624]
        # print(f"真实值长度：{y_true_val.shape}")
        t1, t2 = pd.DataFrame(mlist, columns=['predict']), pd.DataFrame(y_true_val, columns=['true'])

        tt = pd.concat([t1, t2], axis=1)
        pd.DataFrame(tt).to_csv('./pred_data/gcn_transformer_fc/{}/val/第{}次验证集预测值.csv'.format(num,epoch), index=False)



