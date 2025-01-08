# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:37:34 2022

@author: 86136
"""

class UnorderedList(object):
    def __init__(self,onelist=[]):
        self.items=onelist
    
    def append(self,item):
        self.items.append(item)
    
    def slice(self,start,stop):
        list=[]
        for i in range(start-1,stop-1):
            list.append(self.items[i])
        return list

myList=UnorderedList([1,2,3,4])
print(myList.slice(2, 4))