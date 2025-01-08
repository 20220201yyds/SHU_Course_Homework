# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:43:51 2022

@author: 86136
"""

class mylist(list):
    def __init__(self,onelist):
        self.list=onelist
    
    def product(self):
        x=1
        for i in self.list:
            if type(i)!= int :
                return False
            else:
                x*=i
        return x

newlist=mylist([3,4,5,2])
print(newlist.product())
