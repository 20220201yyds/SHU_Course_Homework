# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:16:54 2022

@author: 86136
"""

#找k的过程
def findMink(numlist,k):
    numlist.sort()
    return(numlist[k-1])
    
import random
from timeit import Timer
import matplotlib.pyplot as plt
result=[]#储存列表长度与找出第k小元素的时间
t=Timer("findMink(mylist,k)","from __main__ import mylist,k,findMink")
for i in range(10,101,10):
    mylist=[]
    #构造随机数字列表
    for j in range(i):
        mylist.append(random.randint(1, 100))
    #随机一个k
    k=random.randint(1, 10)
    needtime=t.timeit(number=1000)
    result+=[i,needtime]
#绘图
data1=[]
data2=[]
for item in result:
    data1+=item[0]
    data2+=item[1]
plt.plot(data1,data2)
plt.show()