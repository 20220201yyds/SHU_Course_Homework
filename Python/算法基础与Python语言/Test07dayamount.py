# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 08:40:52 2022

@author: 86136
"""

y_m=input().split()
y=int(y_m[0])
m=int(y_m[1])
if m==2:
    if y%4!=0 or (y%100==0 and y%400!=0):
        print("%d年%d月有28天"%(y,m))
    else:
        print("%d年%d月有29天"%(y,m))
elif m==1 or m==3 or m==5 or m==7 or m==8 or m==10 or m==12:
    print("%d年%d月有31天"%(y,m))
else:
    print("%d年%d月有30天"%(y,m))
    
