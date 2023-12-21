import torch
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

def func(PATH):
    df_num, y_true = create_dataset_number(PATH)
    df_seq = create_dataset_seq(PATH)
    df_seq = torch.tensor([item for item in df_seq]).to(torch.int64)
    tensor_data_num = torch.tensor(df_num, dtype=torch.float32)
    y = torch.tensor([item for item in y_true]).to(torch.float)

    return df_seq, y, y_true,tensor_data_num


