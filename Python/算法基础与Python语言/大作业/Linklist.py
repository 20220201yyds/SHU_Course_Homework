# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:10:34 2022

@author: 86136
"""

class Node():
    def __init__(self,data):
        self.data=data
        self.next=None

class Linklist():
    def __init__(self, node=None):
        self.head = node
        self.length=0
        
    def travel(self):
        cur = self.head
        while cur != None:
            print(cur.data)
            cur = cur.next
        
    def is_empty(self):
        return self.head == None
    
    def append(self,item):       
        node=Node(item)
        if self.is_empty():
            self.head = node
        elif self.length==1:
            if self.head.data<=item:
                self.head.next=node
            else:
                node.next=self.head
                self.head=node
        else:
            if item<self.head.data:
                node.next=self.head
                self.head=node
            else:
                temp1=self.head
                temp2=self.head.next
                while temp2.data<item and temp2.next!=None:
                    temp2=temp2.next
                    temp1=temp1.next
                if temp2.data<item and temp2.next==None:
                        temp2.next=node
                else:
                    node.next=temp2
                    temp1.next=node
        self.length+=1
    
    def pop(self):
        self.head=self.head.next
        self.length-=1


    