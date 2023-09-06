import re
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
# from read import read_pos
from sklearn.preprocessing import scale
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertTokenizer


warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 定义 LSTM 模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.em = nn.Embedding(num_embeddings=27,embedding_dim=128)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(514 , hidden_size)
        self.classifier = nn.Linear( 2*hidden_size , 1)
        self.sig = nn.Sigmoid()

        self.nrom = nn.BatchNorm1d(32)

    def forward(self, x1,x2):

        h0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size).to(device)
        x1 = self.em(x1)
        out, _ = self.lstm(x1, (h0, c0))
        out_lstm = out[:,-1,:]  #1 0.000001
        out_lc = self.fc(x2)
        out_lc = self.nrom(out_lc) # 1.0000000
        output = torch.cat((out_lstm, out_lc), dim=1)
        output = self.classifier(output)
        # output = nn.Sigmoid(output,dim = -1)
        output = self.sig(output)

        return output



class BERTTextClassificationModel(nn.Module):
    def __init__(self, hidden_size, dropout_prob=0.1):
        super(BERTTextClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        for name,param in self.bert.named_parameters():
            # 冻结前面层数
            if   '11' in name:
                param.requires_grad = True # 打开
            else:
                param.requires_grad = False # 冻结
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_size)

        self.fc2 = nn.Linear(133, hidden_size)
        self.classifier = nn.Linear(2*hidden_size, 1)
        self.sig = nn.Sigmoid()

        self.nrom = nn.BatchNorm1d(2)

    def forward(self, input_ids, x2):
        outputs = self.bert(input_ids=input_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)

        out_lc = self.fc2(x2)
        out_lc = self.nrom(out_lc)
        output = torch.cat((logits, out_lc), dim=1)
        output = self.classifier(output)
        # output = nn.Sigmoid(output,dim = -1)
        output = self.sig(output)

        return output


# 定义超参数
input_size = 128
hidden_size = 32
num_layers = 2
output_size = 133
learning_rate = 0.0001
num_epochs = 1
batch_size = 2


def d_loadar(x1,x2,y):
    # 应该在这里   batch size 8
    data = list()
    X = []
    Y = []
    Z = []
    for x,y,z in zip(x1,x2,y):
        X.append(x)
        Y.append(y)
        Z.append(z)
        if len(X) == batch_size:
            o_x = X
            o_y = Y
            o_z = Z
            X = []
            Y = []
            Z = []
            x_res = torch.tensor([item.cpu().detach().numpy() for item in o_x]).cuda()
            y_res = torch.tensor([item.cpu().detach().numpy() for item in o_y]).cuda()
            z_res = torch.tensor([item.cpu().detach().numpy() for item in o_z]).cuda()

            # x_ress = torch.tensor([item.cpu().detach().numpy() for item in [x]]).cuda()
            # y_ress = torch.tensor([item.cpu().detach().numpy() for item in [y]]).cuda()
            # z_ress = torch.tensor([item.cpu().detach().numpy() for item in [z]]).cuda()
            yield (x_res,y_res,z_res)
            # yield (x_ress,y_ress,z_ress)

def create_dataset_number(PATH_x):
    df = pd.read_csv(PATH_x)

    df_num = df.drop(columns=['Year','CycPeptMPDB_ID','Structurally_Unique_ID'
        ,'SMILES','Sequence_LogP','Sequence_TPSA','label']).values
    dfy = pd.read_csv(PATH_x)
    y = dfy['label'].values.reshape(-1,1)
    y = y.astype('float32')
    df_num = scale(df_num)
    return df_num,y


def create_dataset_list(PATH_x):
    df_list = pd.read_csv(PATH_x,usecols=['Sequence_LogP','Sequence_TPSA'])
    df_list['Sequence_LogP'] = df_list['Sequence_LogP'].apply(lambda x: eval(x))
    df_list['Sequence_TPSA'] = df_list['Sequence_TPSA'].apply(lambda x: eval(x))
    a = df_list['Sequence_LogP'].values
    b = df_list['Sequence_TPSA'].values

    max_len = max(len(x) for x in a)
    # print('Sequence_LogP的max_len:',max_len)

    data_padded = np.zeros((len(a), max_len))
    for i, row in enumerate(a):
        data_padded[i, :len(row)] = row

    tensor_data = torch.tensor(data_padded, dtype=torch.float32)
    logp_list = torch.tensor(pad_sequence(tensor_data, batch_first=True, padding_value=0))

    max_len1 = max(len(x) for x in b)
    # print('Sequence_TPSA的max_len:',max_len1)

    data_padded = np.zeros((len(b), max_len))
    for i, row in enumerate(b):
        data_padded[i, :len(row)] = row

    tensor_data = torch.tensor(data_padded, dtype=torch.float32)
    # tensor_data_num = torch.tensor(df_num, dtype=torch.float32)
    tpsa_list = torch.tensor(pad_sequence(tensor_data, batch_first=True, padding_value=0))

    # list_num = torch.stack([logp_list,tpsa_list],dim=0)
    list_num = torch.cat([logp_list, tpsa_list], dim=1)
    # print(type(list_num))
    list_num = scale(list_num)
    list_num = torch.tensor(list_num, dtype=torch.float32)
    # print(type(list_num))

    return list_num

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


def create_dataset_seq(PATH_x):
    df = pd.read_csv(PATH_x,usecols=['SMILES'])
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
    return mlist

