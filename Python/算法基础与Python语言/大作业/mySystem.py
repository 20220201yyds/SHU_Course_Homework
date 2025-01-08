# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:46:59 2022

@author: 86136
"""

import Linklist
myLink=Linklist.Linklist()
k=int(input())
score=int(input())
while score>0:
    if myLink.length<k:
        myLink.append(score)
    else:
        if score>myLink.head.data:
            myLink.append(score)
            myLink.pop()
    print("当前分数线%d"%myLink.head.data)
    score=int(input())