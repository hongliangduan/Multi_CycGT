from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

import pickle
import pandas as pd
import pickle
from tqdm import tqdm

acc = []
recall = []
precision = []
f1 = []

for i in tqdm(range(1,11)):

    X_train=pd.read_csv('../../../data/data_splitClassifier_235/X_train{}.csv'.format(i))
    y_train=pd.read_csv('../../../data/data_splitClassifier_235/y_train{}.csv'.format(i)).to_numpy().reshape(-1)

    test=pd.read_csv('../../../data/data_splitClassifier_235_67/test_{}.csv'.format(i))

    test_x = test.drop(columns=['label'])
    test_y = test['label'].to_numpy()
    numtest = (test_y.shape[0] // 16) * 16
    # 定义KNN模型并进行训练
    knn = KNeighborsClassifier(n_neighbors=100)
    knn.fit(X_train, y_train.astype('int'))

    # 对测试集进行预测并计算准确率
    # y_pred = knn.predict(X_test)
    y_pred = knn.predict_proba(test_x[:numtest])
    y_pred = y_pred[:,1]


    # acc.append(accuracy_score(y_test.astype('int'), y_pred))
    # # print('Accuracy:', acc)
    #
    # f1.append(f1_score(y_test.astype('int'), y_pred))
    # # print('f1:', f1)
    #
    # recall.append(recall_score(y_test.astype('int'), y_pred))
    # # print('recall:', recall)
    #
    # precision.append(precision_score(y_test.astype('int'), y_pred))
    # print('precision:', precision)
    t1,t2 = pd.DataFrame(y_pred,columns=['predict']),pd.DataFrame(test_y[:numtest],columns=['true'])
    tt = pd.concat([t1, t2], axis=1)


    pd.DataFrame(tt).to_csv('./pred_data_67/knn/第{}次测试集预测值.csv'.format(i))
    # 保存模型
    with open("./model_67/knn/knn_model{}.pkl".format(i), "wb") as f:
        pickle.dump(knn, f)

    # 加载模型并进行测试
    # with open("../model/knn/knn_model{}.pkl".format(i), "rb") as f:
    #     knn_loaded = pickle.load(f)
    # y_pred_loaded = knn_loaded.predict(X_test)
    # accuracy_loaded = accuracy_score(y_test.astype('int'), y_pred_loaded)
    # print("Loaded model accuracy:", accuracy_loaded)


# print('Accuracy:', acc)
# print('recall:', recall)
# print('precision:', precision)
# print('f1:', f1)