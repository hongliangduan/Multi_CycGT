


from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':

    for i in tqdm(range(1,11)):

        X_train=pd.read_csv('../../../data/data_splitClassifier_235/X_train{}.csv'.format(i))
        y_train=pd.read_csv('../../../data/data_splitClassifier_235/y_train{}.csv'.format(i)).to_numpy().reshape(-1)

        test=pd.read_csv('../../../data/data_splitClassifier_235/test_{}.csv'.format(i))

        test_x = test.drop(columns=['label'])
        test_y = test['label'].to_numpy()
        numtest = (test_y.shape[0] // 16) * 16

        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(X_train, y_train.astype('int'))

        y_pred = clf.predict_proba(test_x[:numtest])
        y_pred = y_pred[:,1]


        t1,t2 = pd.DataFrame(y_pred,columns=['predict']),pd.DataFrame(test_y[:numtest],columns=['true'])
        tt = pd.concat([t1, t2], axis=1)


        pd.DataFrame(tt).to_csv('./pred_data/rf/experiment_{}_predicted_values.csv'.format(i))
        with open('./model/rf/{}_rf_model_rbf.pkl'.format(i), 'wb') as f:
            pickle.dump(clf, f)

