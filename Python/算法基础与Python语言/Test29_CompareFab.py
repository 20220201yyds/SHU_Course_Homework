# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:02:27 2022

@author: 86136
"""

import time
def Fab_digui(n):
    if n==1 or n==2:
        return 1
    else:
        return Fab_digui(n-1)+Fab_digui(n-2)

def Fab_xun(n):
    if n==1 or n==2:
        return 1
    else:
        Fab=[1,1]
        for i in range(1,n-1):
            Fab.append(Fab[i]+Fab[i-1])
        return Fab.pop()

print(time.process_time())
print(Fab_digui(40))
print(time.process_time())
print(Fab_xun(40))
print(time.process_time())