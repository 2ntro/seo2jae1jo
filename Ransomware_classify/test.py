import csv
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import roc_curve, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np


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
os.chdir('C:/sharedfolder/testmodel')  # 작업할 디렉토리 설정

# 가장먼저 추출한 문자열 들을 가지고 옴
def make_Doc2Vec(software_sentences, ransomware_sentences):
    with open('./test_software_behavior_api_order.csv', 'r', encoding='utf-8') as f:
        lines = csv.reader(f)
        software_sentences = [line for line in lines]
    with open('./test_behavior_api_order.csv', 'r', encoding='utf-8') as f:
        lines = csv.reader(f)
        ransomware_sentences = [line for line in lines]

    features = []  # 만약 추출할게 없다면 비워놔도 됨. 대신 생성은 해놓기.. 추후 수정.

    # doc2vec로 학습을 시키기 위해선 (words , tag 가 필요함)
    tagged_data = []
    # word = ['word1', 'word2', 'word3', 'word4'....], tags = ['software_숫자'] 소프트웨어

    for i, sentence in enumerate(software_sentences):
         tagged_data.append(TaggedDocument(words=featureset_remove_fun(sentence, features)
                                           , tags=['software_' + str(i)]))

    # word = ['word1', 'word2', 'word3', 'word4'....], tags = ['malware_숫자'] 악성코드
    for i, sentence in enumerate(ransomware_sentences):
        tagged_data.append(TaggedDocument(words=featureset_remove_fun(sentence, features)
                                          , tags=['malware_' + str(i)]))

    return software_sentences, ransomware_sentences

# 학습에 사용할 모델 load 시킴
doc2vec_model_name = "../Doc2vec_model_vector30_window15_dm0"
model = Doc2Vec.load(doc2vec_model_name)

software_sentences, ransomware_sentences = make_Doc2Vec(software_sentences=[], ransomware_sentences=[])

software_vector = [model.infer_vector(sentence,alpha=0.025,min_alpha=0.025, epochs=30)
                 for sentence in software_sentences]
ransomware_vector = [model.infer_vector(sentence,alpha=0.025,min_alpha=0.025, epochs=30)
                 for sentence in ransomware_sentences]

software_arrays = np.array(software_vector)
software_labels = np.zeros(len(software_vector)) # 어처피 소프트웨어니까 0으로 초기화 시킬꺼임

ransomware_arrays = np.array(ransomware_vector)
ransomware_labels = np.ones(len(ransomware_vector))  # 어처피 악성코드니까 1으로 초기화 시킬꺼임

test_data = np.vstack((software_arrays,ransomware_arrays))
test_labels = np.hstack((software_labels,ransomware_labels))

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
