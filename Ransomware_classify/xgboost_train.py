import csv
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import numpy as np
import data_process as dp

#이 함수는 API 시퀀스 순서는 보존하되 내가 원하는 API만 제거 후 반환.
def featureset_remove_fun(sentence,remove_features):
    '''
    sentence 문장, remove_features 제거할 특성들. 제거 된 상태로 나가게됨
    '''
    return [e for e in sentence if e not in (remove_features)]

#tscne 그리는 함수임.
def tscne_fun(model_name):
    '''
    model_name 으로 doc2vec 의 모델 이름을 넣어 주면됨.
    '''
    model = Doc2Vec.load(model_name)
    tags = list(model.docvecs.doctags.keys())#dpcvecs에서 태그 데이터 가져옴.
    #software_idx = []
    malware_idx = []
    for i, tag in enumerate(tags):
        #if tag.split('_')[0] == 'software':
         #   software_idx.append(i)#software의 배열 위치를 저장..
        if tag.split('_')[0] == 'malware':
            malware_idx.append(i)#malware의 배열 위치를 저장.
    tsne = TSNE(n_components=2).fit(model.docvecs.doctag_syn0)#2차원으로 변환시킴
    datapoint = tsne.fit_transform(model.docvecs.doctag_syn0)
    fig = plt.figure()#특징 설정
    fig.set_size_inches(40, 20)# 크기 셋팅
    ax = fig.add_subplot(1, 1, 1) #subplot 생성

    #악성코드 그리기. datapoint[malware_idx,0] x좌표,  datapoint[malware_idx,1] y 좌표
    ax.scatter(datapoint[malware_idx,0], datapoint[malware_idx,1],c='r')
    #소프트웨어  그리기
    #ax.scatter(datapoint[software_idx,0], datapoint[software_idx,1],c='b')
    fig.savefig(model_name+'.png')

'''
첫번째!!!!!!!!!!
Doc2vec 형태의 모델 만들기!!!!!!!!!
'''

#train할 모델을 만든뒤 tsne 로 시각화 시킨후 테스트를 할것이다.
os.chdir('C:/Users/tjgml/OneDrive/문서/GitHub/seo2jae1jo/Ransomware_classify/testmodel')#작업할 디렉토리 설정

train_software_path = '../software/behavior_api_order.csv'
train_malware_path = './behavior_api_order.csv'
model_name = "../Doc2vec_model_vector30_window15_dm0"

train_software_sentences, train_malware_sentences = dp.make_Doc2Vec(train_software_path, train_malware_path,
                                                        software_sentences=[], malware_sentences=[])

arrays, labels = dp.make_data(model_name, train_software_sentences, train_malware_sentences)
'''
여기서 부터는 본격적으로 xgboost 데이터 학습시킴
'''
import xgboost as xgb
from sklearn.model_selection import KFold
import collections
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization

xgb.set_config(verbosity=0)
#학습을 튜닝하기전에 먼저 Kfold로 일부를 나누겟음.
kf_test = KFold(n_splits = 5, shuffle = True)
for train_index, test_index in kf_test.split(arrays):
    # split train/validation
    train_data, test_data  = arrays[train_index], arrays[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]
print(len(train_data))
print(len(test_data))

#kFold로 학습시킨후 평균을 내어 반환 하는 함수.
def kFoldValidation(train, features, xgbParams, numRounds, nFolds, target='loss'):#가장 마지막으로 호출됨.
    kf = KFold(n_splits = nFolds, shuffle = True)
    fold_score=[]
    for train_index, test_index in kf.split(train):
        # split train/validation
        X_train, X_valid = train[train_index], train[test_index]
        y_train, y_valid = features[train_index], features[test_index]
        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(xgbParams, dtrain, numRounds, evals = watchlist,early_stopping_rounds = 50,verbose_eval=False)#verbose 옵션으로 나타나지 않게함
        score = gbm.best_score
        fold_score.append(score)
    return np.mean(fold_score)

def xgbCv(train, features, numRounds, eta, gamma, maxDepth, minChildWeight, subsample, colSample):
    # prepare xgb parameters
    params = {
        "objective": "reg:squarederror",
        "booster" : "gbtree",
        "eval_metric": "mae",
        "tree_method": 'auto',
        "silent": 1,
        "eta": eta,
        "max_depth": int(maxDepth),
        "min_child_weight" : minChildWeight,
        "subsample": subsample,
        "colsample_bytree": colSample,
        "gamma": gamma,
        "scale_pos_weight": 0.48#데이터 셋 비율  0/1 소프트웨어 / 악성코드
        }#적용할 파라미터들
    #순서대로 train 학습시킬 데이터, features 특징, 기준이 될 xgb 파라미터, numRounds 반복횟수, nFolds =
    cvScore = kFoldValidation(train, features, params, int(numRounds), nFolds = 5)
    print('CV score: {:.6f}'.format(cvScore))
    return -1.0 * cvScore   # invert the cv score to let bayopt maximize

def bayesOpt(train, features):#가장먼저 호출됨.
    train = train_data
    features = train_labels
    ranges = {
        'numRounds': (1000, 2000),
        'eta': (0.03, 0.1),
        'gamma': (0, 10),
        'maxDepth': (4, 10),
        'minChildWeight': (0, 10),
        'subsample': (0, 1),
        'colSample': (0, 1),
        }#학습에 따라 변경될 값들.
    # proxy through a lambda to be able to pass train and features
    optFunc = lambda numRounds, eta, gamma, maxDepth, minChildWeight, subsample, colSample: xgbCv(train, features, numRounds, eta, gamma, maxDepth, minChildWeight, subsample, colSample)
    bo = BayesianOptimization(optFunc, ranges)

    bo.maximize(init_points = 50, n_iter = 5, kappa = 2, acq = "ei", xi = 0.0)
    best_params = max(bo.res, key=lambda x: x['target'])['params']

    dtrain = xgb.DMatrix(train_data, label=train_labels)
    booster = xgb.train(best_params, dtrain, num_boost_round=1980)

    dtest = xgb.DMatrix(test_data)
    y_pred = booster.predict(dtest)
    y_pred  = y_pred > 0.5
    y_pred = y_pred.astype(int)
    accuracy = accuracy_score(y_pred, test_labels)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))#예측률

    import pickle
    #pickle.dump(booster,open("pima.pickle.dat","wb"))
    loaded_model = pickle.load(open("pima.pickle.dat","rb"))
    y_pred= loaded_model.predict(dtest)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(int)
    accuracy = accuracy_score(y_pred, test_labels)
    print("model Accuracy: %.2f%%" % (accuracy * 100.0))  # 예측률

bayesOpt(arrays,labels)