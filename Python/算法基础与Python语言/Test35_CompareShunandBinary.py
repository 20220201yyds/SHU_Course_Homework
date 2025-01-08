# -*- coding: utf-8 -*-
"""
Created on Fri May  6 22:30:51 2022

@author: 86136
"""
import numpy as np
from timeit import Timer

def binarySearch(alist,item):
    if len(alist)==1:
        if alist[0]==item:
            return True
        else:
            return False
    else:
        midpoint=len(alist)//2
        if item<alist[midpoint]:
            return binarySearch(alist[:midpoint], item)
        else:
            return binarySearch(alist[midpoint+1:], item)
           

alist=np.random.randint(0,1000,size=1000)
print(alist)
print(binarySearch(alist, 100))
t1=Timer("binarySearch(alist,100)","from __main__ import binarySearch,alist")
t2=Timer("list(alist).index(100)","from __main__ import alist")
pasttime1=t1.timeit(1000)
pasttime2=t2.timeit(1000)
print("二分查找:",pasttime1)
print("顺序查找:",pasttime2)