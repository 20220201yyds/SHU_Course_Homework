# -*- coding: utf-8 -*-
"""
Created on Fri May  6 18:05:05 2022

@author: 86136
"""

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
            
if __name__ == '__main__':
    alist=[1,2,3,4,5,6,7]
    print(binarySearch(alist, 9))
