# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:26:02 2022

@author: 17480
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def draw1(data,theta,num,colors):
    x=[[] for i in range(2)]
    for ele in data:
        x[int(ele[-1])].append(ele[:-1])
	#绘制原始数据
    for i in range(num):
        x[i]=np.array(x[i])
        plt.scatter(x[i][:,0],x[i][:,1],color=colors[i])
	#绘制被映射直线
    if theta[0]<0 and theta[1]<0 :
        minus_theta0 = -theta[0]
        minus_theta1 = -theta[1]
    plt.plot([0,minus_theta0],[0,minus_theta1])
	#绘制映射到直线上的点
    for i in range(num):
        for ele in x[i]:
            ta=theta*np.dot(ele,theta)
            plt.plot([ele[0],ta[0]],[ele[1],ta[1]],color=colors[i],linestyle="--")
            plt.scatter(ta[0],ta[1],color=colors[i])
    plt.show()

def c2(data,num):
    n=data.shape[1]-1
    x=[[] for i in range(num)]
    u=[[]for i in range(num)]
    sw=np.zeros([n,n])
    for ele in data:
        print(ele)
        x[int(ele[-1])].append(ele[:-1])
    for i in range(num):
        x[i]=np.array(x[i])
        u[i]=np.mean(x[i],axis=0)
    for i in range(num):
        x[i]=x[i]-u[i]
        sw=sw+np.dot(x[i].T,x[i])
    print("x_i去中心化:\n",x)
    print("计算散度矩阵S_w:\n",sw)
    #计算theta
    theta=np.dot(np.linalg.inv(sw),(u[0]-u[1]).T)
	#单位化
    fm=0
    for i in range(n):
        fm=fm+theta[i]**2
    return theta/np.sqrt(fm)

if __name__ == '__main__':
    data={'density':[0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437,
                     0.666,0.243,0.245,0.343,0.639,0.657,0.36,0.593,0.719],
          'sugar':[0.46,0.376,0.264,0.318,0.215,0.237,0.149,0.211,
                   0.091,0.267,0.057,0.099,0.161,0.198,0.37,0.042,0.103],
          'quality':[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]}
    df=pd.DataFrame(data)
    data=df.values
    colors=['blue','red']
    theta=c2(data,2)
    print("4计算直线向量theta:\n",theta)
    draw1(data,theta,2,colors)