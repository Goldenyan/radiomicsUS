import os
import pandas as pd
import numpy as np
import pymrmr
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn import metrics
import time
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut


#data读取
Path = os.getcwd()
filePath = Path + '\\feature_RF.xlsx'

data = pd.read_excel(filePath)
# data.drop(columns = 'Unnamed: 0', inplace = True)

#方差消除
for colnames,col in data.iteritems() :
    if np.std(data.loc[:,colnames]) == 0 :
        data.drop(columns = colnames, inplace=True)



#添加标签
labelList = []
for each in range(85) :
    if each < 53 :
        labelList.append(0)
    else :
        labelList.append(1)
data.insert(0, 'label',  labelList)
label = data['label']
data.drop(columns = 'label', inplace = True)
columnsName = data.columns

#新建空表用于存放数据
df_looresult = pd.DataFrame(columns=['seed', 'trainACC', 'trainAUC', 'trainSEN','trainSPE', 'traintime', 'valACC', 'testACC', 'testAUC','testSEN','testSPE','testtime'])
df_features = pd.DataFrame()
df_train = pd.DataFrame()
df_test = pd.DataFrame()


loo = LeaveOneOut()
m = 0
seed = 1864
sss = StratifiedShuffleSplit(n_splits = 100, random_state = seed, test_size = 0.2)
for train_index, test_index in sss.split(data, label) :

    df_Ktrain = pd.DataFrame()
    df_Ktest = pd.DataFrame()
    print('----------------------------------------')
    start_traintime = time.time()  # 计时开始

    print('This is the %s times split' % (m + 1))
    print('The test index is', test_index)


    trainData = data.iloc[train_index]
    # Z-score标准化数据
    trainData = trainData.astype(np.float64)
    scaler = StandardScaler().fit(trainData)  # 基于训练集的标准化规则
    trainData = scaler.transform(trainData)  # 将规则应用于训练集
    trainData = pd.DataFrame(trainData)

    testData = data.iloc[test_index]
    testData = testData.astype(np.float64)
    testData = scaler.transform(testData)  # 将规则应用于测试集
    testData = pd.DataFrame(testData)

    trainLabel = label[train_index]
    testLabel = label[test_index]
    trainData.columns = columnsName
    testData.columns = columnsName


    trainData.insert(0, 'label', trainLabel.values)


    # mRMR特征选择
    n=5
    index = pymrmr.mRMR(trainData, 'MID', n)
    df_addFeatures = pd.DataFrame(index)
    df_addFeatures.columns = ['%ssplit' % (m + 1)]
    trainData = trainData[index]
    testData = testData[index]

    # 计算最终保留的特征的相关系数
    # df = data[index]
    # corr = df.corr('spearman').abs()
    # print((np.mean(corr.values) * df.shape[1] - 1) / (df.shape[1] - 1))
    # df_addFeatures.loc[n] = (np.mean(corr.values) * df.shape[1] - 1) / (df.shape[1] - 1)
    # df_features = pd.concat([df_features, df_addFeatures], axis=1)

    # SVM kernal: linear
    # paragrid = {'C': np.logspace(-10, 7, 1000, base=2)}
    # grid = GridSearchCV(svm.SVC(kernel='linear', random_state=1000, tol=1e-4,probability=True), param_grid=paragrid, cv=loo).fit(trainData, trainLabel.values)  # 网格搜索法寻找最优参数
    # cBest = grid.best_params_['C']
    # print('The best parameter C is ', cBest)
    # model = svm.SVC(kernel='linear', C=cBest, random_state=1000, tol=1e-4, probability=True).fit(trainData, trainLabel.values)  # 使用最优参数构建SVM模型

    # Logistic回归分类
    paragrid = {'C': np.logspace(-10, 7, 1000, base=2)}
    grid = GridSearchCV(LogisticRegression(random_state=1000,solver='liblinear', tol=1e-4), param_grid=paragrid, cv=loo, n_jobs=-1).fit(trainData, trainLabel)
    cBest = grid.best_params_['C']
    print('The best parameter C is ', cBest)
    model = LogisticRegression(random_state=1000, C=cBest,solver='liblinear', tol=1e-4).fit(trainData,trainLabel)

    # Random Forest
    # paragrid = {'n_estimators': range(30,70, 1), 'max_depth': [2], 'max_features': [2,3,4], 'max_samples': [0.5],'min_samples_leaf': [6,8,10]}
    # grid = GridSearchCV(RandomForestClassifier(random_state=1000, oob_score=False),
    #                     param_grid=paragrid, cv=loo, n_jobs=-1).fit(trainData, trainLabel)
    # n_estimator = grid.best_params_['n_estimators']
    # max_depth = grid.best_params_['max_depth']
    # max_features = grid.best_params_['max_features']
    # max_samples = grid.best_params_['max_samples']
    # min_samples_leaf = grid.best_params_['min_samples_leaf']
    # # min_samples_split = grid.best_params_['min_samples_split']
    # # max_leaf_nodes = grid.best_params_['max_leaf_nodes']
    # print('The best parameter n_estimators, max_samples and max_features are', n_estimator,'and', max_samples, 'and', max_features, min_samples_leaf)
    # model = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth,max_features=max_features,random_state=1000, oob_score=False,
    #                                max_samples=max_samples,min_samples_leaf=min_samples_leaf).fit(trainData, trainLabel)


    # #Native Bayes
    # grid = GridSearchCV(GaussianNB(), param_grid={'var_smoothing': [1e-9, 1e-8, 1e-7]}, cv=loo, n_jobs=-1).fit(trainData, trainLabel)
    # model = GaussianNB(var_smoothing=grid.best_params_['var_smoothing']).fit(trainData, trainLabel)

    # KNN
    # paragrid = {'n_neighbors': range(10, 30, 1), 'weights': ['uniform']}
    # grid = GridSearchCV(KNeighborsClassifier(), param_grid=paragrid, cv=loo, n_jobs=-1
    #                      ).fit(trainData, trainLabel)
    # n_neighbors = grid.best_params_['n_neighbors']
    # weights = grid.best_params_['weights']
    # print('The best parameter n_neighbors and weights are', n_neighbors, 'and', weights)
    # model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights).fit(trainData, trainLabel)

    score = grid.best_score_
    end_traintime = time.time()  # 计时结束
    traintime = end_traintime - start_traintime  # 训练耗时

    start_testtime = time.time()  # 计时开始

    trainACC = model.score(trainData, trainLabel)
    testACC = model.score(testData, testLabel)
    trainScore = model.predict_proba(trainData)[:, 1]
    testScore = model.predict_proba(testData)[:, 1]

    trainAUC = metrics.roc_auc_score(trainLabel.values, trainScore)
    if trainAUC < 0.5:
        trainAUC = 1 - trainAUC
    trainCM = confusion_matrix(trainLabel.values, model.predict(trainData))
    trainSPE = trainCM[0, 0] / (trainCM[0, 0] + trainCM[0, 1])
    trainSEN = trainCM[1, 1] / (trainCM[1, 0] + trainCM[1, 1])

    testAUC = metrics.roc_auc_score(testLabel.values, testScore)
    if testAUC < 0.5:
        testAUC = 1 - testAUC
    testCM = confusion_matrix(testLabel, model.predict(testData))
    testSPE = testCM[0, 0] / (testCM[0, 0] + testCM[0, 1])
    testSEN = testCM[1, 1] / (testCM[1, 0] + testCM[1, 1])

    end_testtime = time.time()
    testtime = end_testtime - start_testtime  # 测试耗时

    df_Ktrain['Score'] = trainScore
    df_Ktrain['predict'] = model.predict(trainData)
    df_Ktrain['Label'] = trainLabel.values
    df_train = pd.concat([df_train, df_Ktrain], axis=0)

    df_Ktest['Score'] = testScore
    df_Ktest['predict'] = model.predict(testData)
    df_Ktest['Label'] = testLabel.values
    df_test = pd.concat([df_test, df_Ktest], axis=0)

    print('The accuracy of validation are', score)
    print('The prediction is', model.predict(testData))
    print('--------------------------')

    print('The accuracy of train/test data are', trainACC, 'and', testACC)
    print('The AUC of train data is', trainAUC)
    print('The AUC of test data is', testAUC)

    print('The train time of this fold is', traintime)
    print('The test time of this fold is', testtime)
    m = m+1
    df_looresult.loc[m] = [seed, trainACC, trainAUC, trainSEN, trainSPE, traintime, score, testACC, testAUC, testSEN, testSPE, testtime]



savePathlooresults = Path + '\\Kfold\\looresults.xlsx'
savePathfeatures = Path + '\\Kfold\\features.xlsx'
savePathtrainscore = Path + '\\Kfold\\trainscore.xlsx'
savePathtestscore = Path + '\\Kfold\\testscore.xlsx'
df_looresult.to_excel(savePathlooresults)
df_features.to_excel(savePathfeatures)
df_train.to_excel(savePathtrainscore)
df_test.to_excel(savePathtestscore)



