# -*- coding: utf-8 -*-
"""
Created on Fri May  6 22:55:50 2022

@author: 86136
"""

def twoSideBubbleSort(alist):
    exchanges=True;
    passnum=len(alist)-1
    k=0
    while passnum>0 and exchanges:
        exchanges=False
        if k%2==0:
            for i in range(passnum):
                if alist[i]>alist[i+1]:
                    exchanges=True
                    alist[i],alist[i+1]=alist[i+1],alist[i]
            passnum-=1
        else:
            for i in range(passnum):
                if alist[passnum-i-1]>alist[passnum-i]:
                    exchanges=True
                    alist[i],alist[i+1]=alist[i+1],alist[i]
            passnum-=1
        k+=1
    return alist

mylist=[1,3,6,9,3,5,6,2,8]
print(twoSideBubbleSort(mylist))