# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:23:13 2022

@author: 86136
"""

def Pasca_triangleList(n):
    if n==1:
        return [[1]]
    elif n==2:
        return [[1],[1,1]]
    else:
        preTri=Pasca_triangleList(n-1)
        preList=preTri[n-2]
        newList=[1]
        for i in range(0,len(preList)-1):
            newList.append(preList[i]+preList[i+1])
        newList.append(1)
        preTri.append(newList)
        return preTri
    
def Pasca_trianglePrint(Pasca_triList):
    row=len(Pasca_triList)
    for i in range(row):
        for j in range(row-i):
            print("   ",end="")
        for j in range(i+1):
            print("%3d"%Pasca_triList[i][j],end="   ")
        print()

Pasca_triList=Pasca_triangleList(6)
Pasca_trianglePrint(Pasca_triList)