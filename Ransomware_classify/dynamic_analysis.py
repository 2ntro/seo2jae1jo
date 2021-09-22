import os
import json
import glob
import pandas as pd
import ast
import matplotlib.pyplot as plt
import csv
import numpy as np


os.chdir('C:\sharedfolder') #작업하고 있는 디렉토리 변경
os.getcwd()
os.listdir('./cuckoo_json_malware')#현재 폴더의 파일들 목록

class FeatureExtraction:
    '''
    json_file_path에 파일의 경로를 넣어야함
    '''
    def __init__(self,json_file_path):
        json_data=open(json_file_path).read()
        data = json.loads(json_data)#json 형태로 변환
        self.data = data
        self.md5 = self.get_md5(data)#md5가져오기 . md5 = 최종 파일이름.
        self.pe_imports = self.get_pe_imports_name(data)#pe_imports 특징 추출
        self.apistats = self.get_behavior_apistats(data)#apistats 특징 추출
        self.get_behavior_api_order = self.get_behavior_api_order(data)
        self.apistats_dataframe = self.apistats_dataframe()
        self.pe_imports_dataframe = self.pe_imports_dataframe()

    '''
    만약 새로운 feature 을 추출하고 싶으면 아래 get 함수 처럼 따라서 추가 시켜주면됨.
    '''
    #pefile header import 추출
    def get_pe_imports_name(self,data):
        result_list = []
        try:
            for pe_imports in data['static']['pe_imports']:
                for import_api in pe_imports['imports']:
                    result_list.append(import_api['name'])
        except Exception as e:
            pass
        return result_list#[apiname1, apiname2, ...]

    #apistatas 추출함. (api,호출 횟수)
    def get_behavior_apistats(self,data):#없는 파일도 있으니 꼭 try 해야함.
        result_list = []
        try:
            for process in data['behavior']['apistats']: # 어떤 process 가 나올지 모르니..
                for apistats, num in data['behavior']['apistats'][process].items():
                    result_list.append((apistats,num))
                #break#첫번째 process 만 추출
        except Exception as e:
            pass
        return result_list#[(apiname1,3), (apiname2,1), ...]

    def get_behavior_api_order(self,data):
        result_list = []
        try:
            for process in data['behavior']['processes']: # 어떤 process 가 나올지 모르니..
                for call in process['calls']:
                    result_list.append(call['api'])
            return result_list
        except Exception as e:
            pass
        return result_list

    def get_md5(self,data):
        return data['target']['file']['md5']

        '''
    def make_csv(self,folder_name):#저장할 폴더 이름, py 위치에서 생성
        디렉토리 생성하는 try  except 문
        try:#없을경우 생성, 있을경우 그냥 넘어감.
            if not(os.path.isdir(folder_name)):
                os.makedirs(os.path.join(folder_name))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise
        #make_dataframe()
        '''
    def pe_imports_dataframe(self):
        try:
            self.pe_imports
            #row_api_counts = np.reshape(api_counts,(1,-1))
            return pd.DataFrame(1,dtype='int',columns = self.pe_imports, index=[self.md5])#데이터 프레임생성
        except Exception as e:
            return pd.DataFrame(index=[self.md5])#apistats가 비어있을 경우 index 만 추출


    def apistats_dataframe(self):
        '''
        input_dataframe 과 class에서 생성한 datafrmae을 합친 뒤 반환.
        return type: DataFrame(인덱스 = md5name, 열 = apiname , 내용 = 호출 횟수)
        '''
        try:
            api_names,api_counts = zip(*self.apistats)
            row_api_counts = np.reshape(api_counts,(1,-1))
            return pd.DataFrame(row_api_counts,dtype='int',columns = api_names, index=[self.md5])#데이터 프레임생성
        except Exception as e:
            return pd.DataFrame(index=[self.md5])#apistats가 비어있을 경우 index 만 추출




'''
일단 malware sample이 호출하는 모든 api 데이터들을 한줄로 모아야한다.
apistats의 데이터프레임을 추출한뒤 csv파일로 저장 / 이건 일종의 순서쌍이지
'''
folder_path = './cuckoo_json_malware/' #추출할 폴더
df = pd.DataFrame()#저장할 데이터 프레임 생성
for input_folder in glob.glob(folder_path+'*'):#현재 폴더의 모든 파일이름 추출
    for input_file in glob.glob(input_folder+'./*'):
        print(input_file)
        feature = FeatureExtraction(input_file)
        df = feature.apistats_dataframe
        df.to_csv("apistats.csv",mode='a')


'''
pe_imports의 데이터프레임을 추출한뒤 csv파일로 저장
'''

folder_path = './cuckoo_json_malware/' #추출할 폴더
df = pd.DataFrame()#저장할 데이터 프레임 생성
for input_folder in glob.glob(folder_path+'*'):#현재 폴더의 모든 파일이름 추출
    for input_file in glob.glob(input_folder+'./*'):
        print(input_file)
        feature = FeatureExtraction(input_file)
        df = feature.apistats_dataframe
        df.to_csv("pe_imports.csv", mode='a')


#csv 파일 만들기  /software_feature/filename.csv

'''
api 호출 순서대로 나열 된 리스트 파일
만약 추출이 안된 [] 비어있는 리스트라면 저장 하지 않음.
추후 Doc2Vec에 사용될 csv 파일들.
'''
folder_path = './cuckoo_json_malware/' #추출할 파일들을 가져오는폴더

f = open('behavior_api_order_set.csv', 'w', encoding='utf-8', newline='')
api_call_total = []

for input_folder in glob.glob(folder_path+'*'):#현재 폴더의 모든 파일이름 추출
    for input_file in glob.glob(input_folder+'./*'):#현재 폴더의 모든 파일이름 추출
        feature = FeatureExtraction(input_file)
        if len(feature.get_behavior_api_order) == 0:
            continue
        else:
            #wr = csv.writer(f)
            behavior_api_order_set = set(feature.get_behavior_api_order)
            behavior_api_order_set = list(behavior_api_order_set)
            api_call_total.append(behavior_api_order_set)
            #wr.writerow(behavior_api_order_set)

api_call_total = sum(api_call_total, [])
api_call_total = set(api_call_total)
print(len(api_call_total))

f = open('behavior_api_order.csv', 'w', encoding='utf-8', newline='')
for input_file in glob.glob(folder_path+'*'):#현재 폴더의 모든 파일이름 추출
    feature = FeatureExtraction(input_file)
    if len(feature.get_behavior_api_order) == 0:
        continue
    else:
        wr = csv.writer(f)
        wr.writerow(feature.get_behavior_api_order)
f.close()
