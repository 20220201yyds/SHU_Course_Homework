# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 19:37:53 2022

@author: 86136
"""

import turtle
PART_OF_PATH=1
Tried=2
Obstacle='+'
DeadEnd="-"

class Maze:
    def __init__(self,maze_filename):
        rows_in_maze=0
        columns_in_maze=0
        self.maze_list=[]
        maze_file=open(maze_filename,"r",encoding="utf-8")
        for line in maze_file:
            rowList=[]
            col=0
            for ch in line[:-1]:
                rowList.append(ch)
                if ch=='S':
                    self.start_row=rows_in_maze
                    self.start_col=col
                col+=1
            rows_in_maze+=1
            self.maze_list.append(rowList)
            columns_in_maze=len(rowList)
            
        self.rows_in_maze=rows_in_maze
        self.columns_in_maze=columns_in_maze
        self.x_translate=-columns_in_maze/2
        self.y_translate=rows_in_maze/2
        self.t=turtle.Turtle()
        self.t.shape('turtle')
        self.wn=turtle.Screen()
        self.wn.setworldcoordinates(-(columns_in_maze-1)/2-.5,-(rows_in_maze-1)/2-.5,
                                    (columns_in_maze-1)/2+0.5, (rows_in_maze-1)/2+.5)
        print(self.maze_list)
        
    def draw_maze(self):
        self.t.speed(10000)
        for y in range(self.rows_in_maze):
            for x in range(self.columns_in_maze):
                if self.maze_list[y][x]==Obstacle:
                    self.draw_wall(x+self.x_translate,-y+self.y_translate,'orange')
        self.t.fillcolor('green')
        
    def draw_wall(self,x,y,color):
        self.t.up()
        self.t.goto(x-.5,y-.5)
        self.t.color(color)
        self.t.fillcolor(color)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for i in range(4):
            self.t.forward(1)
            self.t.right(90)
        self.t.end_fill()
        self.t.end_fill()
    
    def move_turtle(self,x,y):
        self.t.up()
        self.t.setheading(self.t.towards(x+self.x_translate,-y+self.y_translate))
        self.t.goto(x+self.x_translate,-y+self.y_translate)
        
    def drop_bread(self,color):
        print("draw")
        self.t.dot(10,color)

        
    def update_position(self,row,col,val=None):
        if val:
            self.maze_list[row][col]=val
        self.move_turtle(col, row)
        
        if val==PART_OF_PATH:
            color='green'
        elif val==Obstacle:
            color='red'
        elif val==Tried:
            color='blue'

        elif val==DeadEnd:
            color='red'
        else:
            color=None
        
        if color:
            self.drop_bread(color)
    
    def is_exit(self,row,col):
        return (row==0 or row==self.rows_in_maze-1 or col==0 or col==self.columns_in_maze-1)
    
    def __getitem__(self,idx):
        return self.maze_list[idx]
    
def search_from(maze,start_row,start_col):
    maze.update_position(start_row,start_col)
    print(start_row,start_col)
    if maze.maze_list[start_row][start_col]==Obstacle:
        return False
    if maze.maze_list[start_row][start_col]==Tried or maze.maze_list[start_row][start_col]==DeadEnd:
        return False
    if maze.is_exit(start_row, start_col):
        maze.update_position(start_row, start_col,PART_OF_PATH)
        return True
    maze.update_position(start_row, start_col,Tried)
    found=search_from(maze,start_row-1,start_col) or \
          search_from(maze,start_row,start_col-1) or \
          search_from(maze,start_row+1,start_col) or \
          search_from(maze,start_row,start_col+1)
    if found:
        maze.update_position(start_row, start_col,PART_OF_PATH)
    else:
        maze.update_position(start_row, start_col,DeadEnd)
    return found
    
my_maze=Maze('maze.txt')
my_maze.draw_maze()
my_maze.update_position(my_maze.start_row, my_maze.start_col)
search_from(my_maze,my_maze.start_row,my_maze.start_col)
        