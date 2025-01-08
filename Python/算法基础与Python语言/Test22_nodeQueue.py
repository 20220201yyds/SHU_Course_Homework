# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 14:17:58 2022

@author: 86136
"""
class Node(object):
    def __init__(self,inputvalue=None,inputnext=None):
        self.value=inputvalue
        self.next=inputnext
    
    def getValue(self):
        return self.value
    
    def getNext(self):
        return self.next

    def setValue(self,new_value):
        self.value = new_value

    def setNext(self,new_next):
        self.next = new_next
        
class nodeQueue(object):   
    def __init__(self):
        self.head=Node()
        self.tail=None
        self.length=0
    
    def enqueue(self,value):
        newNode=Node(value,None)
        if self.length==0:
            self.head=newNode
            self.tail=newNode
        if self.length==1:
            self.tail=newNode
            self.head.setNext(newNode)
        else:
            self.tail.setNext(newNode)
            self.tail=newNode
        self.length+=1
        
    def dequeue(self):
        if self.length>1:
            value=self.head.getValue()
            self.head=self.head.getNext()
        elif self.length==1:
            self.Head=Node()
            self.tail=None
        else:
            return False
        self.length-=1
        return value
    
    
    
if __name__ == '__main__':
    myQueue=nodeQueue()
    myQueue.enqueue(1)
    myQueue.enqueue("2")
    myQueue.enqueue(3)
    print(myQueue.dequeue())
    print(myQueue.dequeue())
