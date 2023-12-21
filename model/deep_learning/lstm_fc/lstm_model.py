
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
import re
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
from sklearn.preprocessing import scale

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.em = nn.Embedding(num_embeddings=27,embedding_dim=128)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(103 , hidden_size)
        self.classifier = nn.Linear( 2*hidden_size , 1)
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        self.nrom = nn.BatchNorm1d(32)

    def forward(self, x1,x2):

        h0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size).to(device)
        x1 = self.em(x1)
        out, _ = self.lstm(x1, (h0, c0))
        out_lstm = out[:,-1,:]
        out_lc = self.fc(x2)
        out_lc = self.nrom(out_lc)
        output = torch.cat((out_lstm, out_lc), dim=1)
        output = self.classifier(output)
        output = self.sig(output)

        return output

def d_loadar(x1,x2,y):
    X = []
    Y = []
    Z = []
    for x,y,z in zip(x1,x2,y):
        X.append(x)
        Y.append(y)
        Z.append(z)
        if len(X) == 8:
            o_x = X
            o_y = Y
            o_z = Z
            X = []
            Y = []
            Z = []
            x_res = torch.tensor([item.cpu().detach().numpy() for item in o_x]).cuda()
            y_res = torch.tensor([item.cpu().detach().numpy() for item in o_y]).cuda()
            z_res = torch.tensor([item.cpu().detach().numpy() for item in o_z]).cuda()

            yield (x_res,y_res,z_res)


def create_dataset_number(PATH_x,PATH_y):
    df = pd.read_csv(PATH_x)
    df_num = df.drop(columns=['Year','CycPeptMPDB_ID','Structurally_Unique_ID'
        ,'SMILES','Sequence_LogP','Sequence_TPSA','label']).values

    dfy = pd.read_csv(PATH_y)
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
    data_padded = np.zeros((len(a), max_len))
    for i, row in enumerate(a):
        data_padded[i, :len(row)] = row
    tensor_data = torch.tensor(data_padded, dtype=torch.float32)
    logp_list = torch.tensor(pad_sequence(tensor_data, batch_first=True, padding_value=0))
    data_padded = np.zeros((len(b), max_len))
    for i, row in enumerate(b):
        data_padded[i, :len(row)] = row

    tensor_data = torch.tensor(data_padded, dtype=torch.float32)
    tpsa_list = torch.tensor(pad_sequence(tensor_data, batch_first=True, padding_value=0))
    list_num = torch.cat([logp_list, tpsa_list], dim=1)
    list_num = scale(list_num)
    list_num = torch.tensor(list_num, dtype=torch.float32)
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
