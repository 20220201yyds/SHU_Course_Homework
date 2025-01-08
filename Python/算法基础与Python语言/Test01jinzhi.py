# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:56:18 2022

@author: 86136
"""
numstr=input()
if numstr.count(".")==1:
    numlist=numstr.split(".")
    intnum=int(numlist[0])#整数部分
    floatnum=int(numlist[1])#小数部分
    num=intnum+10**(-len(str(floatnum)))*floatnum
    pointnum=num-intnum
    #转置整数部分
    outintstr=""
    while intnum!=0:
        x=intnum%2
        intnum=intnum//2
        outintstr+=str(x)
    outintstr=outintstr[::-1]
    
    #转置小数部分
    outfloatstr=""
    for i in range(0,8):
        pointnum*=2
        if pointnum>1:
            pointnum-=1
            outfloatstr+="1"
        elif pointnum==1:
            outfloatstr+="1"
            break
        else:
            outfloatstr+="0"
    print("输入数为%s"%numstr)
    print("2进制为0b%s.%s"%(outintstr,outfloatstr))
elif numstr.count(".")==0:
    num=int(numstr)
    print("输入数为%d\n2进制为%s"%(num,bin(num)))


    
