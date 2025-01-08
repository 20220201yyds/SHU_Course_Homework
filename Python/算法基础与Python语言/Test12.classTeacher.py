# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:43:37 2022

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
    
class Teacher(People):
    def __init__(self, namestr, citystr,schoolstr):
        super().__init__(namestr,citystr)
        self.school=schoolstr
    
    def __str__(self):
        return "[name=%s , city=%s , school=%s]" % (self.name,self.city,self.school)
        
    def newschool(self,newschool):
        self.school=newschool
        
    def __lt__(self,other):
        return self.school < other.school
s=[]
s.append(Teacher("zhang","Shanghai","SHU"))
s.append(Teacher("li","Hangzhou","SJTU"))
s.append(Teacher("wang","Beijing","FD"))
s.append(Teacher("zeng","Guangzhou","HKU"))
s.sort()
for item in s:
    print(item,end="")