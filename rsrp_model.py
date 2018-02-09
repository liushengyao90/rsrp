#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split

#处理训练集，测试集
#处理训练集，测试集
train_data1 = pd.read_excel('train_new.xlsx')
test_data1 = pd.read_excel('test_new.xlsx')
train_data1.fillna(0,inplace = True)
test_data1.fillna(0,inplace = True)

train_data1_1 = train_data1['参考信号接收功率采样点点数'] != 0
new_train_data1 = train_data1[train_data1_1]

test_data1_1 = test_data1['参考信号接收功率采样点点数'] != 0
new_test_data1 = test_data1[test_data1_1]

for i in new_train_data1.columns[4:52]:
    new_train_data1[i]/= new_train_data1['参考信号接收功率采样点点数']
for i in new_test_data1.columns[4:52]:
    new_test_data1[i]/= new_test_data1['参考信号接收功率采样点点数']

new_test_data1.to_excel('test1.xlsx')
new_train_data1.to_excel('train1.xlsx')


train_data = pd.read_excel('train1.xlsx')
test_data = pd.read_excel('test1.xlsx')

# 构建训练测试数据
# 特征处理
feat_names = train_data.columns[4:52].tolist()
X_train = train_data[feat_names].values
print('共有{}维特征。'.format(X_train.shape[1]))
X_test = test_data[feat_names].values


# 标签处理
train_labels = train_data['网元中文名'].values
test_labels = test_data['网元中文名'].values

# 使用sklearn.preprocessing.LabelEncoder进行类别标签处理
from sklearn.preprocessing import LabelEncoder

label_enc = LabelEncoder()
y_train = label_enc.fit_transform(train_labels)
y_test = label_enc.transform(test_labels)

print('类别标签：', label_enc.classes_)
for i in range(len(label_enc.classes_)):
    print('编码 {} 对应标签 {}。'.format(i, label_enc.inverse_transform(i)))



from sklearn.model_selection import GridSearchCV

def train_model(X_train, y_train, X_test, y_test, model_config, cv_val=3):
    """
        返回对应的最优分类器及在测试集上的准确率
    """
    model = model_config[0]
    parameters = model_config[1]

    if parameters is not None:
        # 需要调参的模型
        clf = GridSearchCV(model, parameters, cv=cv_val, scoring='accuracy')
        clf.fit(X_train, y_train)
        print('最优参数：', clf.best_params_)
        print('验证集最高得分： {:.3f}'.format(clf.best_score_))
    else:
        # 不需要调参的模型，如朴素贝叶斯
        model.fit(X_train, y_train)
        clf = model

    test_acc = clf.score(X_test, y_test)
    print('测试集准确率：{:.3f}'.format(test_acc))
    return clf, test_acc


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model_dict = {'kNN':    (KNeighborsClassifier(),        {'n_neighbors': [5, 10, 15]}),
              'LR':     (LogisticRegression(),          {'C': [0.01, 1, 100]}),
              'SVM':    (SVC(),                         {'C': [100, 1000, 10000]}),
              'DT':     (DecisionTreeClassifier(),      {'max_depth': [50, 100, 150]}),
              'RF':     (RandomForestClassifier(),      {'n_estimators': [100, 150, 200]}),
              'GBDT':   (GradientBoostingClassifier(),  {'learning_rate': [0.1, 1, 10]})}


results_df = pd.DataFrame(columns=['Accuracy (%)','Time (s)'], index=list(model_dict.keys()))
results_df.index.name = 'Model'
models = []

results_df = pd.DataFrame(columns=['Accuracy (%)'], index=list(model_dict.keys()))
results_df.index.name = 'Model'
models = []

for model_name, model_config in model_dict.items():
    print('训练模型：', model_name)
    start =time.time()
    model, acc = train_model(X_train, y_train,
                          X_test, y_test,
                          model_config)
    end = time.time()
    duration = end - start
   
    print('耗时{:.4f}s '.format(duration), end=', ')
    models.append(model)
    results_df.loc[model_name] = acc * 100
    print('\n')

# 保存结果
results_df.to_csv('./pred_results1.csv')

results_df.plot(kind='bar')
plt.ylabel('Accuracy (%)')
plt.tight_layout()
plt.savefig('./pred_results1.png')
plt.show()


# 保存最优模型
import pickle

best_model_idx = results_df.reset_index()['Accuracy (%)'].argmax()
best_model = models[best_model_idx]

saved_model_path = './predictor1.pkl'
with open(saved_model_path, 'wb') as f:
    pickle.dump(best_model, f)
