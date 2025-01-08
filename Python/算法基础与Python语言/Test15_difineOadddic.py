# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:41:09 2022

@author: 86136
"""

from timeit import Timer
t1=Timer('dic.get(100)',"from __main__ import dic")
t2=Timer('dic[1]=1',"from __main__ import dic")
dic={}
for i in range(10,201,10):
    dic[i]=i*2
    pastTime1=t1.timeit(number=1000)
    pastTime2=t2.timeit(number=1000)
    print("字典长度为%d时，增加项时间为：%.8f取值键为100项时间为：%.8f"%(i,pastTime2,pastTime1))