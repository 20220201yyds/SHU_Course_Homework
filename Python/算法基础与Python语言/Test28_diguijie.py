# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:59:58 2022

@author: 86136
"""

def jie(n):
    if n==1:
        return 1
    else:
        return(n*jie(n-1))
    
print(jie(10))