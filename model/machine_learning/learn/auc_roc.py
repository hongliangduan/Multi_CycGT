
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

for i in range(1,11):
    dt = pd.read_csv('./pred_data/dt/第{}次测试集预测值.csv'.format(i))
    svm = pd.read_csv('./pred_data/svm/第{}次测试集预测值.csv'.format(i))
    rf = pd.read_csv('./pred_data/rf/第{}次测试集预测值.csv'.format(i))
    knn = pd.read_csv('./pred_data/knn/第{}次测试集预测值.csv'.format(i))



    predict_dt,true_dt = dt['predict'].to_numpy().reshape(-1,1),dt['true'].to_numpy().reshape(-1,1)
    predict_svm,true_svm = svm['predict'].to_numpy().reshape(-1,1),svm['true'].to_numpy().reshape(-1,1)
    predict_rf,true_rf = rf['predict'].to_numpy().reshape(-1,1),rf['true'].to_numpy().reshape(-1,1)
    predict_knn,true_knn = knn['predict'].to_numpy().reshape(-1,1),knn['true'].to_numpy().reshape(-1,1)




    # 计算 ROC 曲线和 AUC 值
    fpr_dt, tpr_dt, thresholds_dt = roc_curve(true_dt, predict_dt)
    roc_auc_dt = auc(fpr_dt, tpr_dt)


    fpr_rf, tpr_rf, thresholds_rf = roc_curve(true_rf, predict_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    print(roc_auc_rf)
    fpr_svm, tpr_svm, thresholds_svm = roc_curve(true_svm, predict_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    
    fpr_knn, tpr_knn, thresholds_knn = roc_curve(true_knn, predict_knn)
    roc_auc_knn = auc(fpr_knn, tpr_knn)
    # 绘制 ROC 曲线图
    plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='CNN (AUC = %0.3f)' % roc_auc)

    plt.plot(fpr_dt, tpr_dt, color='red', lw=2, label='dt (AUC = %0.3f)' % roc_auc_dt)
    plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label='rf (AUC = %0.3f)' % roc_auc_rf)
    plt.plot(fpr_svm, tpr_svm, color='green', lw=2, label='svm (AUC = %0.3f)' % roc_auc_svm)
    plt.plot(fpr_knn, tpr_knn, color='pink', lw=2, label='knn (AUC = %0.3f)' % roc_auc_knn)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('./imgs/第{}个测试集的图.png'.format(i))
    plt.show(block=False)
    plt.clf()
