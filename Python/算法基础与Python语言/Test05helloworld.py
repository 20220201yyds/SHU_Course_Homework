# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:25:25 2022

@author: 86136
"""

print("姓名：",end="")
name=input()
print("分数（以空格间隔）：",end="")
scorelist=input().split()
all=0
for i in range(0,3):
    all+=int(scorelist[i])
print("Hello World, Hello %s, Your average score is %.1f"%(name,all/3))