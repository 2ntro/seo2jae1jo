import csv

import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


def featureset_remove_fun(sentence,remove_features):
    '''
    sentence 문장, remove_features 제거할 특성들. 제거 된 상태로 나가게됨
    '''
    return [e for e in sentence if e not in (remove_features)]

def make_Doc2Vec(software_path, ransom_path, software_sentences, malware_sentences):
    with open(software_path, 'r', encoding='utf-8') as f:
        lines = csv.reader(f)
        software_sentences = [line for line in lines]
    with open(ransom_path, 'r', encoding='utf-8') as f:
        lines = csv.reader(f)
        malware_sentences = [line for line in lines]

    features = []# 만약 추출할게 없다면 비워놔도 됨. 대신 생성은 해놓기.. 추후 수정.

    #doc2vec로 학습을 시키기 위해선 (words , tag 가 필요함)
    tagged_data =[]
    #word = ['word1', 'word2', 'word3', 'word4'....], tags = ['software_숫자'] 소프트웨어

    for i,sentence in enumerate(software_sentences):
           tagged_data.append(TaggedDocument(words = featureset_remove_fun(sentence,features)
                                          , tags = ['software_'+str(i)]))

    #word = ['word1', 'word2', 'word3', 'word4'....], tags = ['malware_숫자'] 악성코드
    for i,sentence in enumerate(malware_sentences):
        tagged_data.append(TaggedDocument(words = featureset_remove_fun(sentence,features)
                                          , tags = ['malware_'+str(i)]))

    return software_sentences, malware_sentences

def make_data(model_name, software_sentences, malware_sentences):
    doc2vec_model_name = model_name
    model = Doc2Vec.load(doc2vec_model_name)

    #software_sentences, malware_sentences = make_Doc2Vec(software_sentences=[], malware_sentences=[])

    #software문자열을 가져온뒤 학습 doc model에 넣고
    software_vector = [model.infer_vector(sentence,alpha=0.025,min_alpha=0.025, epochs=30)
                     for sentence in software_sentences]
    malware_vector = [model.infer_vector(sentence,alpha=0.025,min_alpha=0.025, epochs=30)
                     for sentence in malware_sentences]

    software_arrays = np.array(software_vector)
    software_labels = np.zeros(len(software_vector)) # 어처피 소프트웨어니까 0으로 초기화 시킬꺼임

    malware_arrays = np.array(malware_vector)
    malware_labels = np.ones(len(malware_vector)) # 어처피 악성코드니까 1으로 초기화 시킬꺼임


    #데이터 셋 합치기.
    data = np.vstack((software_arrays,malware_arrays))
    labels = np.hstack((software_labels,malware_labels))

    return data, labels