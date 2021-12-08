# -*- coding: utf-8 -*-
"""
    :author: XieJava
    :url: http://ishareread.com
    :copyright: © 2021 XieJava <xiejava@ishareread.com>
    :license: MIT, see LICENSE for more details.
"""
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from DataUtils import getTokens,modelfile_path,vectorfile_path

#从文件中获取数据集
def getDataFromFile(filename='data/data.csv'):
    input_url = filename
    data_csv = pd.read_csv(input_url, ',', error_bad_lines=False)
    data_df = pd.DataFrame(data_csv)
    url_df = np.array(data_df)
    random.shuffle(url_df)
    y = [d[1] for d in url_df]
    inputurls = [d[0] for d in url_df]
    return inputurls,y


#训练,通过逻辑回归模型训练
def trainLR(datapath):
    all_urls,y=getDataFromFile(datapath)
    url_vectorizer = TfidfVectorizer(tokenizer=getTokens)
    x = url_vectorizer.fit_transform(all_urls)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    l_regress = LogisticRegression()                  # Logistic regression
    l_regress.fit(x_train, y_train)
    l_score = l_regress.score(x_test, y_test)
    print("score: {0:.2f} %".format(100 * l_score))
    return l_regress,url_vectorizer

#训练，通过SVM支持向量机模型训练
def trainSVM(datapath):
    all_urls, y = getDataFromFile(datapath)
    url_vectorizer = TfidfVectorizer(tokenizer=getTokens)
    x = url_vectorizer.fit_transform(all_urls)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    svmModel=svm.LinearSVC()
    svmModel.fit(x_train, y_train)
    svm_score=svmModel.score(x_test, y_test)
    print("score: {0:.2f} %".format(100 * svm_score))
    return svmModel,url_vectorizer

#保存模型及特征
def saveModel(model,vector):
    #保存模型
    file1 = modelfile_path
    with open(file1, 'wb') as f:
        pickle.dump(model, f)
    f.close()
    #保存特征
    file2 = vectorfile_path
    with open(file2, 'wb') as f2:
        pickle.dump(vector, f2)
    f2.close()

if __name__ == '__main__':
    #model,vector=trainLR('data/data.csv')
    model, vector = trainSVM('data/data.csv')
    saveModel(model,vector)