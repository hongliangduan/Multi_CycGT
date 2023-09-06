


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import pickle
import pandas as pd
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

    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, y_train.astype('int'))


    # 预测测试集
    # y_pred = clf.predict(X_test)
    y_pred = clf.predict_proba(test_x[:numtest])
    y_pred = y_pred[:,1]
    print(y_pred)

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
    # # print('precision:', precision)
    t1,t2 = pd.DataFrame(y_pred,columns=['predict']),pd.DataFrame(test_y[:numtest],columns=['true'])
    tt = pd.concat([t1, t2], axis=1)


    pd.DataFrame(tt).to_csv('./pred_data_67/rf/第{}次测试集预测值.csv'.format(i))
    # 保存模型
    with open('./model_67/rf/第{}次rf_model_rbf.pkl'.format(i), 'wb') as f:
        pickle.dump(clf, f)

# print('Accuracy:', acc)
# print('recall:', recall)
# print('precision:', precision)
# print('f1:', f1)