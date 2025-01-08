# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 08:53:01 2022

@author: 86136
"""

s_n=input().split()
n=int(s_n[1])   
str=s_n[0]
str_1=str[0:len(str)-n:]
str_2=str[len(str)-n::]
print(str_2+str_1)

