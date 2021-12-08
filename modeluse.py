# -*- coding: utf-8 -*-
"""
    :author: XieJava
    :url: http://ishareread.com
    :copyright: © 2021 XieJava <xiejava@ishareread.com>
    :license: MIT, see LICENSE for more details.
"""
from flask import Flask
import pickle
from DataUtils import modelfile_path,vectorfile_path

app = Flask(__name__)

#载入已经训练好的模型
def loadModel():
    file1 = modelfile_path
    with open(file1, 'rb') as f1:
        model = pickle.load(f1)
    f1.close()

    file2 = vectorfile_path
    with open(file2, 'rb') as f2:
        vector = pickle.load(f2)
    f2.close()
    return model,vector

#通过接口进行调用
@app.route('/<path:path>')
def show_predict(path):
    X_predict = []
    X_predict.append(path)
    model, vector = loadModel()
    x = vector.transform(X_predict)
    y_predict = model.predict(x)
    print(y_predict[0])
    return "url predict: "+str(y_predict[0])

if __name__ == "__main__":
	app.run()