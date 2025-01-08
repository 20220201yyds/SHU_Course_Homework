# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 16:21:33 2022

@author: 86136
"""
#排队等待洗车问题：不定时来车，每辆车洗10分钟(程序为10s)
import random
import schedule
class Car(object):
    def __init__(self):
        self.carindex=random.randint(1, 99)
        print("%dcar construct"%self.carindex)
    def getindex(self):
        return self.carindex

class Station(object):
    def __init__(self):
        self.items=[]
        print("construct succeed")
    def carcoming(self):
        car=Car()
        self.items.insert(0,car)
        print("No.%d car come"%car.getindex())
    def carLeaving(self):
        if len(self.items)>=1:
            print("No.%d car leave"%self.items.pop().getindex())

def job1():
    global mystation
    print("using come")
    mystation.carcoming()
def job2():
    global mystation
    print("using leave")
    mystation.carLeaving()
    
if __name__ == '__main__':
    mystation=Station()
    schedule.every(1).to(3).seconds.do(mystation.carcoming)
    schedule.every(2).seconds.do(mystation.carLeaving)
    while True:
        schedule.run_pending()
        