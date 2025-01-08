# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 09:03:14 2022

@author: 86136
"""

def judgeFlower(num):
    numstr=str(num)
    m=len(numstr)
    sum=0
    for i in range(0,m):
        sum+=int(numstr[i])**m
    if sum==num:
        return True
    else:
        return False
    
max=int(input())
for i in range (100,max+1):
    if judgeFlower(i):
        print(i)