# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:11:00 2022

@author: 86136
"""

s=input()
x=0
print(len(s))
if s[0]=="1" and len(s)==11:
    for i in s:
        if "0">i or i>"9":
            print(s+"不合法")
            x=1
            break
    if x==0:
        print(s+"是合法电话号码")
else:
    print(s+"不合法")