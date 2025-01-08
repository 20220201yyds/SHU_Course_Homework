# -*- coding: utf-8 -*-
"""
Created on Fri May 27 12:06:38 2022

@author: 86136
"""

class BinarySearchTree:
    def __init__(self):
        self.root=None
        self.size=0
    
    def length(self):
        return self.size
    
    def __iter__(self):
        return self.root.__iter__()
    
    def put(self,key,val):
        if self.root:
            self._put(key,val,self.root)
        else:
            self.root=TreeNode(key, val)
        self.size+=1
    
    def _put(self,key,val,currentNode):
        if key<currentNode.key:
            if currentNode.hasLeftChild():
                self._put(key, val, currentNode.leftChild)
            else:
                currentNode.leftChild=TreeNode(key,val,currentNode)
        else:
            if currentNode.hasRightChild():
                self._put(key,val,currentNode.rightChild)
            else:
                currentNode.rightChild=TreeNode(key,val,currentNode)

class TreeNode:
    def __init__(self,key,val,left=None,right=None,parent=None):
        self.key=key
        self.val=val
        self.leftChild=left
        self.rightChild=right
        self.parent=parent
        
    def hasLeftChild(self):
        return self.leftChild
    
    def hasRightChild(self):
        return self.rightChild
    
    def isLeftChild(self):
        return self.parent and self.parent.leftChild==self
    
    def isRightChild(self):
        return self.parent and self.parent.rightChild==self
    
    def isRoot(self):
        return not self.parent
    
    def isLeaf(self):
        return not(self.rightChild or self.leftChild)
    
if __name__ == '__main__':
    tree=BinarySearchTree()
    for i in range(1,11):
        tree.put(i,i)
    
    