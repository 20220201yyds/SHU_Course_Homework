# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 20:32:57 2022

@author: 86136
"""

import string

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

#直接转换
def midToBack(midstr):
    prec={"*":3,"/":3,"+":2,"-":2,"(":1}
    opStack=Stack()
    strStack=Stack()
    tokenList=midstr.split( )
    
    for token in tokenList:
        if token in string.ascii_uppercase:
            strStack.push(token)
        elif token=="(":
            opStack.push(token)
        elif token==")":
            topToken=opStack.pop()
            while topToken!="(":
                temp1=strStack.pop()
                temp2=strStack.pop()
                tempstr=temp2+temp1+topToken
                strStack.push(tempstr)
                topToken=opStack.pop()
        else:
            if opStack.isEmpty() or prec[token]>prec[opStack.peek()]:
                opStack.push(token)
            else:
                topToken=opStack.pop()
                temp1=strStack.pop()
                temp2=strStack.pop()
                tempstr=temp2+temp1+topToken
                strStack.push(tempstr)
                opStack.push(token)
    
    while not opStack.isEmpty():
        topToken=opStack.pop()
        temp1=strStack.pop()
        temp2=strStack.pop()
        tempstr=temp2+temp1+topToken
        strStack.push(tempstr)       
    return " ".join(strStack.pop())

str1="( A + B ) * ( C + D ) * ( E + F )"
str2="A + ( ( B + C ) * ( D + E ) )"
str3="A * B * C * D + E + F"
print(midToBack(str1))
print(midToBack(str2))
print(midToBack(str3))