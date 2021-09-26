import csv
import os
import data_process as dp
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
os.chdir('C:/Users/tjgml/OneDrive/문서/GitHub/seo2jae1jo/Ransomware_classify/testmodel')#작업할 디렉토리 설정
train_software_path = '../software/behavior_api_order.csv'
train_malware_path = './behavior_api_order.csv'
test_software_path = './test_software_behavior_api_order.csv'
test_malware_path = './test_behavior_api_order.csv'
model_name = "../Doc2vec_model_vector30_window15_dm0"

def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    return accuracy, precision, recall, confusion

train_software_sentences, train_malware_sentences = dp.make_Doc2Vec(train_software_path, train_malware_path,
                                                        software_sentences=[], malware_sentences=[])
test_software_sentences, test_malware_sentences = dp.make_Doc2Vec(test_software_path, test_malware_path,
                                                        software_sentences=[], malware_sentences=[])

x_train, y_train = dp.make_data(model_name, train_software_sentences, train_malware_sentences)
x_test, y_test = dp.make_data(model_name, test_software_sentences, test_malware_sentences)

rf_clf = RandomForestClassifier()
rf_clf.fit(x_train, y_train)
y_pred = rf_clf.predict(x_test)

acc, precision, recall, _ = get_clf_eval(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (acc * 100.0))  # 예측률
print("Precision : %.2f%%" % (precision * 100.0))  # 정밀도
print("Recall : %.2f%%" % (recall * 100.0))  # 재현율
print("f1 score : %.2f%%" % (f1 * 100.0))  # f1_score

