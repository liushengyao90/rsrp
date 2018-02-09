#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time

#打开文件，选出校园室分数据
file_name = input("请输入文件名：")
data_file = pd.read_excel(file_name)
station_data = pd.read_excel('station_data_1.xlsx')
new_data_end = pd.merge(data_file,station_data,how='inner',left_on='网元中文名',right_on='扇区中文名')
new_data_end.fillna(0,inplace = True)
only_data = new_data_end['参考信号接收功率采样点点数'] != 0
new_data = new_data_end[only_data]
for i in new_data.columns[4:52]:
    new_data[i]/= new_data['参考信号接收功率采样点点数']



#特征处理

feat_names = new_data.columns[4:52].tolist()
X_data = new_data[feat_names].values



import pickle


saved_model_path = './predictor.pkl'
# 加载保存的模型
with open(saved_model_path, 'rb') as f:
    predictor = pickle.load(f)

#进行预测
pred_result = predictor.predict(X_data)
# pred_genre = label_enc.inverse_transform(pred_result)
print('预测类型：', pred_result)
new_data['预测结果'] = pred_result

new_data.to_excel('0104_out1.xlsx')

