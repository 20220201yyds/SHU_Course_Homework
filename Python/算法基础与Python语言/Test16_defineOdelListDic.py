# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:51:17 2022

@author: 86136
"""

from timeit import Timer
mydic={}
t1=Timer("del(mylist)","from __main__ import mylist")
t2=Timer("del(mydic)","from __main__ import mydic")
for i in range(100):
    mylist=list(range(i))
    mydic[i]=i
    delListTime=t1.timeit(number=1)
    delDicTime=t2.timeit(number=1)
    print("删除长度都为%d的列表和字典，需要的时间分别为%.10f,%.10f"%(i,delListTime,delDicTime))