# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 09:50:31 2022

@author: 17480
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))

def load():
    data={'density':[0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437,
                     0.666,0.243,0.245,0.343,0.639,0.657,0.36,0.593,0.719],
          'sugar':[0.46,0.376,0.264,0.318,0.215,0.237,0.149,0.211,
                   0.091,0.267,0.057,0.099,0.161,0.198,0.37,0.042,0.103],
          'quality':[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]}
    df=pd.DataFrame(data)
    return df

def evaluate(x_train,y_train,w):
    data=np.mat(x_train).astype(float)

    y=sigmoid(np.dot(data,w))
    b,c=np.shape(y)#功能是查看矩阵或者数组的维数。
    rightcount=0

    for i in range(b):
        flag=-1
        if y[i,0]>0.5:
            flag=1
        elif y[i,0]<0.5:
            flag=0
        if y_train[i] == flag:
            rightcount+=1

    rightrate = rightcount/len(x_train)
    return rightrate
    
def training(df):
    x_train = df.iloc[:,:-1]
    y_train = df.iloc[:,-1]
    data = np.mat(x_train).astype(float)
    label = np.mat(y_train).transpose()
    w=np.ones((x_train.shape[1],1))
    
    step=0.01
    num=1
    threshold=0.8 #数据集共17组数据，至少预测正确14组
    rightrate=0
    while rightrate<threshold:
        y_pre = sigmoid(np.dot(data,w))
        loss = y_pre - label
        change = np.dot(np.transpose(data),loss)
        w = w - change * step
        rightrate = evaluate(x_train,y_train,w)
        if num%20 == 0:
            print(str(num)+"次正确率为：%.4f"%rightrate)
        num += 1
    else:
        print(str(num)+"次正确率为：%.4f"%rightrate)
    return w
    
     
if  __name__ == '__main__':
    df = load()
    w = training(df)
    print(w)
    x_train = df.iloc[:,:-1]
    data=np.mat(x_train).astype(float)
    y=sigmoid(np.dot(data,w))
    plt.subplot(121)
    plt.scatter(df['density'].loc[df['quality']==1],
                df['sugar'].loc[df['quality']==1],c='red')
    plt.scatter(df['density'].loc[df['quality']==0],
                df['sugar'].loc[df['quality']==0],c='blue')
    plt.subplot(122)
    x=np.arange(0,0.8,step=0.01)
    plt.plot(x,(-x*w[0,0]/w[1,0]))
    for i in range(len(y)):
        if y[i,0]>0.5:
            plt.scatter(df['density'].iloc[i],
                        df['sugar'].loc[i],c='red')
        else:
            plt.scatter(df['density'].iloc[i],
                        df['sugar'].loc[i],c='blue')
    plt.show()
    