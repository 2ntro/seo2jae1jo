import csv
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import roc_curve, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import data_process as dp


def roc_curve_plot(y_test, pred_proba_c1):
    # 임계값에 따른 FPR, TPR 값
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)
    # ROC 곡선을 시각화
    plt.plot(fprs, tprs, label='ROC')
    # 가운데 대각선 직선
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    # FPR X 축의 scla 0.1 단위 지정
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()

# 이 함수는 API 시퀀스 순서는 보존하되 내가 원하는 API만 제거 후 반환.
def featureset_remove_fun(sentence, remove_features):
    '''
    sentence 문장, remove_features 제거할 특성들. 제거 된 상태로 나가게됨
    '''
    return [e for e in sentence if e not in (remove_features)]

def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    return accuracy, precision, recall, confusion
'''
첫번째!!!!!!!!!!
Doc2vec 형태의 모델 만들기!!!!!!!!!
'''

# train할 모델을 만든뒤 tsne 로 시각화 시킨후 테스트를 할것이다.
os.chdir('C:/Users/tjgml/OneDrive/문서/GitHub/seo2jae1jo/Ransomware_classify/testmodel')#작업할 디렉토리 설정
test_software_path = './test_software_behavior_api_order.csv'
test_malware_path = './test_behavior_api_order.csv'
model_name = "../Doc2vec_model_vector30_window15_dm0"


test_software_sentences, test_malware_sentences = dp.make_Doc2Vec(test_software_path, test_malware_path,
                                                        software_sentences=[], malware_sentences=[])
test_data, test_labels = dp.make_data(model_name, test_software_sentences, test_malware_sentences)

import xgboost as xgb
from sklearn.model_selection import KFold
import collections
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization

def test_model(test, features):  # 가장먼저 호출됨.

    dtest = xgb.DMatrix(test_data)
    import pickle
    loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
    y_pred = loaded_model.predict(dtest)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(int)

    accuracy, precision, recall, _ = get_clf_eval(test_labels, y_pred)

    f1 = f1_score(test_labels, y_pred)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))  # 예측률
    print("Precision : %.2f%%" % (precision * 100.0)) #정밀도
    print("Recall : %.2f%%" % (recall * 100.0)) #재현율
    print("f1 score : %.2f%%" % (f1 * 100.0)) #f1_score

    roc_curve_plot(test_labels, y_pred)

test_model(test_data, test_labels)
