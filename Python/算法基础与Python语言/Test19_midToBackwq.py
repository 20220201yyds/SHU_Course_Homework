# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 20:32:57 2022

@author: 86136
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 19:14:25 2022

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

#完全括号
def midToBack(midString):
    while(midString.count(" ")!=0):
        midString=midString.replace(" ","")
    indexs = list()
    entities = list()
    
    i = 0
    while i<len(midString):
        if midString[i] in "+-*/":
            if midString[i] in "*/":
                tmp = entities[len(entities)-1]
                entities.remove(tmp)
                res = ""
                opStack = Stack()
                for j in midString[i+1:]: 
                    if j == "(":
                        opStack.push(i)
                    elif j ==")":
                        opStack.pop()
                    res += j
                    if opStack.isEmpty():
                        break
                nextent = res
                entities.append("("+tmp+midString[i]+nextent+")")
                i+=(len(nextent)+1)
            elif midString[i] in "+-":
                indexs.append(midString[i])
                i+=1
        else:
            res = ""
            opStack = Stack()
            for j in midString[i:]: 
                if j == "(":
                    opStack.push(i)
                elif j ==")":
                    opStack.pop()
                res += j
                if opStack.isEmpty():
                    break
            tmp = res
            entities.append(tmp)
            i += len(tmp) 
            
    while len(entities)>1:
        entities[0] = "(" + entities[0] + indexs[0] + entities[1] +")"
        indexs.remove(indexs[0])
        entities.remove(entities[1])
    
    midString = entities[0]
    result = []
    indexstack = Stack()
    for i in range(len(midString)):
        if midString[i] == "(":
            result.append("_")
        elif midString[i] in "+-*/":
            result.append("_")
            indexstack.push(midString[i])
        elif midString[i] == ")":
            result.append(indexstack.pop())
        else:
            result.append(midString[i])
    while(result.count("_")):
        result.remove("_")
    return " ".join(result)

str1="( A + B ) * ( C + D ) * ( E + F )"
str2="A + ( ( B + C ) * ( D + E ) )"
str3="A * B * C * D + E + F"
print(midToBack(str1))
print(midToBack(str2))
print(midToBack(str3))