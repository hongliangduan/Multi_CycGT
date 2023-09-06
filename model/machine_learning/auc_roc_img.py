from sklearn.metrics import roc_curve, auc, \
    accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import pandas as pd

dt_acc = []
dt_f1 = []
dt_recall = []

mapper = {
    accuracy_score: [[], []],
    f1_score: [[], []],
    recall_score: [[], []],
    precision_score: [[], []],
    'auc': [[],[],],
}

names = ['gcn_transfrom_fc' ]


acc_list= []
for i in range(1, 4):
    # gcn_transfrom_fc = pd.read_csv('./pred_data/gcn_transformer_fc/10/test/第{}次测试集预测值.csv'.format(i))
    # gcn_transfrom_fc = pd.read_csv('./pred_data/gcn_transformer_fc/10/val/第{}次验证集预测值.csv'.format(i))
    gcn = pd.read_csv('./pred_data/gcn/第{}次测试集预测值.csv'.format(i))

    # predict,true = gcn_transfrom_fc['predict'],gcn_transfrom_fc['true']
    predict_gcn,true_gcn = gcn['predict'],gcn['true']




    # print(accuracy_score(true,predict))
    # acc_i = accuracy_score(true,predict)
    # acc_list.append(acc_i)
    print(accuracy_score(true_gcn,predict_gcn))

    predicts = [
        [], []
    ]
    # csvs = [gcn_transfrom_fc ]
    # csvs = [gcn_transfrom_fc , gcn]

    # for i in range(len(csvs)):
    #     predict, true = csvs[i]['predict'].to_numpy().reshape(-1, 1), csvs[i]['true'].to_numpy().reshape(-1, 1)
        # threshold = 0.5
        # predict = [1 if y >= threshold else 0 for y in predict]
        # predicts[i].extend([predict, true])
    #
    # for k in mapper.keys():
    #     if k == 'auc':
    #         continue
    #     for j in range(1):
    #         mapper[k][j].append(k(predicts[j][0], predicts[j][1]))
    #
    # # mapper['auc'] = [[], [], [], [],[],[],[],[],[]]
    # for j in range(1):
    #     fpr_dt, tpr_dt, thresholds_dt = roc_curve(predicts[j][1], predicts[j][0])
    #     # roc_auc_dt = auc(fpr_dt, tpr_dt)
    #     mapper['auc'][j].append(auc(fpr_dt, tpr_dt))
    #

# acc_list.sort(reverse=True)
print(acc_list)


for k in mapper.keys():
    k_name = k if isinstance(k, str) else k.__name__
    # k_name = k.__name__
    print(k_name)
    table = {}
    for i in range(len(names)):
        table[k_name + '_' + names[i]] = mapper[k][i]

        pd.DataFrame(table).to_csv('{}.csv'.format(k_name))


