# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:16:32 2022

@author: 86136
"""

import BinaryHeap
scoreHeap=BinaryHeap.BinaryHeap()
k=int(input())
score=int(input())
while score>0:                             # n倍
    if scoreHeap.length()<k:
        scoreHeap.insert(score)            # append为1，向上重整二叉堆为log2k
    else:
        if score>scoreHeap.rootValue():
            scoreHeap.insert_pop(score)    # 替换根节点为1，向下重整二叉堆为log2k
    print("当前分数线%d"%scoreHeap.rootValue())
    score=int(input())