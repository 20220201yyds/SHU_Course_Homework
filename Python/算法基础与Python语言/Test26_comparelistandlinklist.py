# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 21:10:37 2022

@author: 86136
"""
from timeit import Timer
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
        
class LinkList(object):   
    def __init__(self):
        self.head=Node()
        self.tail=None
        self.length=0
    
    def append(self,value):
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
        
    def delete(self,index):
        if self.length==0:
            print("it is empty")
            return
        elif index<0 or index>=self.length:
            print("out of range")
            return 
        elif index==0:
            self.head=self.head.getNext()
            self.length-=1
        j = 0
        node=self.head
        prev=self.head
        while node.next and j < index:
            prev=node
            node=node.next
            j+=1
        if j==index:
            prev.next=node.next
            self.length-=1
        return 

if __name__ == '__main__':
    norlist=[]
    mylist=LinkList()
    t1=Timer("norlist.append(1)","from __main__ import norlist")
    t2=Timer("mylist.append(1)","from __main__ import mylist")
    norlist=list(range(10000))
    for i in range(1,10001):
        mylist.append(i)
    t3=Timer("norlist.pop(0)","from __main__ import norlist")
    t4=Timer("mylist.delete(0)","from __main__ import mylist")
    print("list append time:%.8f"%t1.timeit(number=1000))
    print("Linklist append time:%.8f"%t2.timeit(number=1000))
    print("list pop time:%.8f"%t3.timeit(number=1000))
    print("Linklist delete time:%.8f"%t4.timeit(number=1000))
    