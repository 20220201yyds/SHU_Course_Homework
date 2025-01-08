# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:07:38 2022

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

#中缀直接转换为后缀表达式
def midCalculate(midstr):
    prec={"*":3,"/":3,"+":2,"-":2,"(":1}
    opStack=Stack()
    numStack=Stack()
    letterpre=""
    for letter in midstr:  
        if letter >='0' and letter <='9':
            if  '0' <= letterpre <= '9':
                temp=numStack.pop()
                temp=temp*10+int(letter)
                numStack.push(temp)
            else:
                numStack.push(int(letter))
        elif letter=="(":
            opStack.push(letter)
        elif letter==")":
            topToken=opStack.pop()
            while topToken!="(":
                temp1=numStack.pop()
                temp2=numStack.pop()
                if topToken=="+":
                    tempResult=temp2+temp1
                elif topToken=="-":
                    tempResult=temp2-temp1
                elif topToken=="*":
                    tempResult=temp2*temp1
                else:
                    if temp1==0:
                        return 0
                    else:
                        tempResult=temp2/temp1
                numStack.push(tempResult)
                topToken=opStack.pop()
        else:
            if opStack.isEmpty() or prec[letter]>prec[opStack.peek()]:
                opStack.push(letter)
            else:
                topToken=opStack.pop()
                temp1=numStack.pop()
                temp2=numStack.pop()
                if topToken=="+":
                    tempResult=temp2+temp1
                elif topToken=="-":
                    tempResult=temp2-temp1
                elif topToken=="*":
                    tempResult=temp2*temp1
                else:
                    if temp1==0:
                        return False
                    else:
                        tempResult=temp2/temp1
                numStack.push(tempResult)
                opStack.push(letter)
        letterpre=letter
    
    while not opStack.isEmpty():
        topToken=opStack.pop()
        temp1=numStack.pop()
        temp2=numStack.pop()
        if topToken=="+":
            tempResult=temp2+temp1
        elif topToken=="-":
            tempResult=temp2-temp1
        elif topToken=="*":
            tempResult=temp2*temp1
        else:
            if temp1==0:
                return False
            else:
                tempResult=temp2/temp1
        numStack.push(tempResult)       
    return numStack.pop()

    
if __name__ == '__main__':
    s=input()
    print(midCalculate(s))

    
    