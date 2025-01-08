# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 15:52:09 2022

@author: 86136
"""

class People:
    def __init__(self,namestr,citystr):
        self.name,self.city=namestr,citystr
      
    def moveto(self,newcity):
        self.city=newcity
    
    def __lt__(self,other):
        return self.city < other.city
    
    def __str__(self):
        return "[name=%s , city=%s]" % (self.name,self.city)
s=[]
s.append(People("zhang","Shanghai"))
s.append(People("li","Hangzhou"))
s.append(People("wang","Beijing"))
s.append(People("zeng","Guangzhou"))
s.sort()
for item in s:
    print(item,end="")
