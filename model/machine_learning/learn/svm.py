

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import pandas as pd
from tqdm import tqdm
import pickle

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
    # 训练 SVM 模型
    clf = svm.SVC(kernel='rbf',probability=True)
    clf.fit(X_train, y_train.astype('int'))

    # 预测测试集
    # y_pred = clf.predict(X_test)
    y_pred = clf.predict_proba(test_x[:numtest])
    y_pred = y_pred[:,1]
    print(y_pred)


    t1,t2 = pd.DataFrame(y_pred,columns=['predict']),pd.DataFrame(test_y[:numtest],columns=['true'])
    tt = pd.concat([t1, t2], axis=1)


    pd.DataFrame(tt).to_csv('./pred_data_67/svm/第{}次测试集预测值.csv'.format(i))
    with open('./model_67/svm/第{}次svm_model_rbf.pkl'.format(i), 'wb') as f:
        pickle.dump(clf, f)


# print('Accuracy:', acc)
# print('recall:', recall)
# print('precision:', precision)
# print('f1:', f1)