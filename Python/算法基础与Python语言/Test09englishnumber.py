# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 08:57:24 2022

@author: 86136
"""

numdic={"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
        "six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11}
s=input()
x=0
for item in numdic.keys():
    if item==s:
        x=1
        print(numdic.get(item))
if x==0:
    print("Not Found")
