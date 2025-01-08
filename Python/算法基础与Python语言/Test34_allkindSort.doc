# -*- coding: utf-8 -*-
"""
Created on Fri May  6 18:23:30 2022

@author: 86136
"""

def bubbleSort(alist):
    for i in range(len(alist)-1):
        for j in range(len(alist)-1-i):
            if alist[j]>alist[j+1]:
                temp=alist[j],alist[j+1]=alist[j+1],alist[j]
    return alist
        
def SelectionSort(alist):
    newlist=[]
    for i in range(len(alist)):
        maxitem=alist[0]
        for item in alist:
            if item>maxitem:
                maxitem=item
        alist.remove(maxitem)
        newlist.insert(0, maxitem)
    return newlist
        
def InsertionSort(alist):
    newlist=[]
    for item in alist:
        if len(newlist)==0:
            newlist.append(item)
        else:
            for i in range(len(newlist)):
                if newlist[i]>item:
                    newlist.insert(i,item)
                    break
                if i==len(newlist)-1:
                    newlist.append(item)
    return newlist

def ShellSort(alist):
    sublistcount=len(alist)//2
    while(sublistcount>0):
        for startposition in range(sublistcount):
            for i  in range(startposition+sublistcount,len(alist),sublistcount):
                currentValue=alist[i]
                position=i
                while position>=sublistcount and alist[position-sublistcount]>currentValue:
                    alist[position]=alist[position-sublistcount]
                    position=position-sublistcount
                alist[position]=currentValue
        sublistcount=sublistcount//2
    return alist

def MergeSort(alist):
    if len(alist)<=1:
        return alist
    middle=len(alist)//2
    left=MergeSort(alist[:middle])
    right=MergeSort(alist[middle:])
    
    merged=[]
    while left and right:
        if left[0] <= right[0]:
            merged.append(left.pop(0))
        else: 
            merged.append(right.pop(0))
    merged.extend(right if right else left)
    return merged

def QuickSort(alist):
    return quickSortHelper(alist,0,len(alist)-1)
def quickSortHelper(alist,first,last):
    if first<last:
        pivovalue=alist[first]
        leftmark=first+1
        rightmark=last
        done=False
        while not done:
            while leftmark<=rightmark and alist[leftmark]<=pivovalue:
                leftmark+=1
            while alist[rightmark]>=pivovalue and rightmark>=leftmark:
                rightmark-=1
            if rightmark<leftmark:
                done=True
            else:
                alist[leftmark],alist[rightmark]=alist[rightmark],alist[leftmark]
        alist[first],alist[rightmark]=alist[rightmark],alist[first]
        splitpoint=rightmark       
        quickSortHelper(alist, first, splitpoint-1)
        quickSortHelper(alist, splitpoint+1, last)
    return alist

alist=['P',"Y",'T','H','O','N']
print(bubbleSort(alist))
alist=['P',"Y",'T','H','O','N']
print(SelectionSort(alist))
alist=['P',"Y",'T','H','O','N']
print(InsertionSort(alist))
alist=['P',"Y",'T','H','O','N']
print(ShellSort(alist))
alist=['P',"Y",'T','H','O','N']
print(MergeSort(alist))
alist=['P',"Y",'T','H','O','N']
print(QuickSort(alist))