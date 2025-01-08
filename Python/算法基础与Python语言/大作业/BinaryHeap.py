# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:25:52 2022

@author: 86136
"""

class BinaryHeap:
    def __init__(self):
        self.heapList=[0]
        self.size=0
    
    def length(self):
        return self.size
    
    def rootValue(self):
        if len(self.heapList) ==1:
            print("二叉堆没有元素")
            return 
        else:
            return self.heapList[1]
        
    def show(self):
        print(self.heapList)
        
    def insert(self,newkey):
        self.heapList.append(newkey)
        self.size+=1
        self.percUp(self.size)
    
    def percUp(self,i):
        while i//2>0:
            if self.heapList[i]<self.heapList[i//2]:
                self.heapList[i],self.heapList[i//2]=self.heapList[i//2],
                self.heapList[i]
            i=i//2
    
    def insert_pop(self,newkey):
        self.heapList[1]=newkey
        self.rootDown()
        
    def rootDown(self):
        i=1
        while i<=self.size/2:
            if i==self.size/2:#正好位于只有一个子节点的节点处
                if self.heapList[i]>self.heapList[2*i]:
                    self.heapList[i],self.heapList[2*i]=self.heapList[2*i],self.heapList[i]
                break
            if self.heapList[i]>min(self.heapList[2*i],self.heapList[2*i+1]):
                if self.heapList[2*i]>self.heapList[2*i+1]:
                    self.heapList[i],self.heapList[2*i+1]=self.heapList[2*i+1],self.heapList[i]
                    i=2*i+1
                else:
                    self.heapList[i],self.heapList[2*i]=self.heapList[2*i],self.heapList[i]
                    i=2*i