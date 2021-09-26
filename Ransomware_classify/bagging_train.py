from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import numpy as np
import csv
import os
import data_process as dp

def featureset_remove_fun(sentence,remove_features):
    '''
    sentence 문장, remove_features 제거할 특성들. 제거 된 상태로 나가게됨
    '''
    return [e for e in sentence if e not in (remove_features)]

'''
첫번째!!!!!!!!!!
Doc2vec 형태의 모델 만들기!!!!!!!!!
'''
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    return accuracy, precision, recall, confusion

#train할 모델을 만든뒤 tsne 로 시각화 시킨후 테스트를 할것이다.
os.chdir('C:/Users/tjgml/OneDrive/문서/GitHub/seo2jae1jo/Ransomware_classify/testmodel')#작업할 디렉토리 설정

train_software_path = '../software/behavior_api_order.csv'
train_malware_path = './behavior_api_order.csv'
test_software_path = './test_software_behavior_api_order.csv'
test_malware_path = './test_behavior_api_order.csv'
model_name = "../Doc2vec_model_vector30_window15_dm0"

train_software_sentences, train_malware_sentences = dp.make_Doc2Vec(train_software_path, train_malware_path,
                                                        software_sentences=[], malware_sentences=[])

data, labels = dp.make_data(model_name, train_software_sentences, train_malware_sentences)

lr_clf = LogisticRegression()
knn_clf = KNeighborsClassifier()

vo_clf = VotingClassifier(estimators=[('LR', lr_clf),
                                      ('KNN', knn_clf)],
                          voting='soft')

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

vo_clf.fit(x_train, y_train)
y_pred = vo_clf.predict(x_test)
acc, precision, recall, _ = get_clf_eval(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (acc * 100.0))  # 예측률
print("Precision : %.2f%%" % (precision * 100.0))  # 정밀도
print("Recall : %.2f%%" % (recall * 100.0))  # 재현율
print("f1 score : %.2f%%" % (f1 * 100.0))  # f1_score

