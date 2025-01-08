# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:23:22 2022

@author: 86136
"""

class Stack(object):
    def __init__(self):
         self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items)-1]

    def size(self):
        return len(self.items)

def HTMLMatch(s):
    stack=Stack()
    for x in s:
        if x=="<":
            flag=0
            temp=""
            temp+=x
        elif x ==">":
            temp+=x
            if flag:
                if stack.isEmpty()==False:
                    temp=temp.replace("/","")
                    if temp==stack.peek():
                        stack.pop()
            else:
                stack.push(temp)
            temp=""
        else:
            if x=="/":
                flag=1
            temp+=x
    if stack.isEmpty():
        return 1
    else:
        return 0

f=open("D:\\tt\\college\\必修课\\Python\\算法基础与Python语言\\html_01.html",encoding='utf-8')
html=f.read()
if HTMLMatch(html):
    print("True")
else:
    print("False")