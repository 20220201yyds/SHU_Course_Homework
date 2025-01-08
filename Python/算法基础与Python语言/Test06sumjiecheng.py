# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 08:38:14 2022

@author: 86136
"""

import math
n=int(input())
sum=0
for i in range(1,n+1):
    sum+=math.factorial(i)
print(sum)