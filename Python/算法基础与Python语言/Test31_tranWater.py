# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:53:33 2022

@author: 86136
"""
import sys
kettle_big = 0
kettle_small = 0
capacity_big = 4
capacity_small = 3
target_status = 2
 
# 已经存在过的状态
old_status = [[0,0]]
# back
back_kettle_big = 0
back_kettle_small = 0
 
def back_kettle():
    global back_kettle_small
    global back_kettle_big
    back_kettle_big = kettle_big
    back_kettle_small = kettle_small
 
def kettle_status():
    global kettle_big
    global kettle_small
    status = [kettle_big, kettle_small]
    return status

def full_big():
    global capacity_big
    global kettle_big
    back_kettle()
    kettle_big = capacity_big
    check_kettle()
 
def full_small():
    global capacity_small
    global kettle_small
    back_kettle()
    kettle_small = capacity_small
    check_kettle()
 
def clear_big():
    global kettle_big
    back_kettle()
    kettle_big = 0
    check_kettle()
 
def clear_small():
    global kettle_small
    back_kettle()
    kettle_small = 0
    check_kettle()
 
def small_to_big():
    global kettle_big
    global kettle_small
    global capacity_big
    back_kettle()
    kettle_small = kettle_big + kettle_small - capacity_big
    kettle_big = capacity_big
    check_kettle()
 
def big_to_small():
    global kettle_big
    global kettle_small
    global capacity_small
    back_kettle()
    kettle_big = kettle_small + kettle_big - capacity_small
    kettle_small = capacity_small
    check_kettle()
 
def small_to_big_full():
    global kettle_big
    global kettle_small
    back_kettle()
    kettle_big = kettle_small + kettle_big
    kettle_small = 0
    check_kettle()
 
def big_to_small_full():
    global kettle_small
    global kettle_big
    back_kettle()
    kettle_small = kettle_big + kettle_big
    kettle_big = 0
    check_kettle()
 
 
# 检查是否查找过
def check_kettle():
    global kettle_big
    global kettle_small
    if kettle_status() in old_status:
        kettle_big = back_kettle_big
        kettle_small = back_kettle_small
 
    else:
        if kettle_big == target_status:
            print("\nFind answer!")
            # print(old_status,end='')
            old_status.append(kettle_status())
            for i in old_status:
                print(i,end='')
                print("-->",end='')
            print("Finish!")
            sys.exit(0)
        else:
            old_status.append(kettle_status())
            find_status()
 
 
# 递归查找存在可能使kettle_big = 2的情况，当查到任意一种情况后终止
def find_status():
    global kettle_small
    global kettle_big
    if kettle_big < capacity_big or kettle_status() == [0,0]:
        full_big()
    if kettle_small < capacity_small:
        full_small()
    if kettle_big > 0:
        clear_big()
    if kettle_small > 0:
        clear_small()
    if kettle_small + kettle_big >= capacity_big and kettle_small > 0:
        small_to_big()
    if kettle_small + kettle_big >= capacity_small and kettle_big > 0:
        big_to_small()
    if kettle_small + kettle_big <= capacity_big and kettle_small > 0:
        small_to_big_full()
    if kettle_small + kettle_big <= capacity_small and kettle_big > 0:
        big_to_small_full()
 
 
if __name__ == '__main__':
    find_status()
    