# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:32:20 2022

@author: 86136
"""

from timeit import Timer
t1=Timer('x[100]',"from __main__ import x")
for i in range(2000,100001,2000):
    x=list(range(i))
    pastTime=t1.timeit(number=1000)
    print("数组长度为%d时，索引第100项时间为：%.8f"%(i,pastTime))