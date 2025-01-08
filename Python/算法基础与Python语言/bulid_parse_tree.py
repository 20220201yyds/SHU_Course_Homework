
from pythonds.basic import Stack
from pythonds.trees import BinaryTree
import operator
import string
def str_repair(midstr):
    prec={"*":3,"/":3,"+":2,"-":2,"(":1}
    opStack=Stack()
    strStack=Stack()
    tokenList=midstr  
    for token in tokenList:
        if token in "0123456789":
            strStack.push(token)
        elif token=="(":
            opStack.push(token)
        elif token==")":
            topToken=opStack.pop()
            while topToken!="(":
                temp1=strStack.pop()
                temp2=strStack.pop()
                tempstr="("+temp2+topToken+temp1+")"
                strStack.push(tempstr)
                topToken=opStack.pop()
        else:
            if opStack.isEmpty() or prec[token]>prec[opStack.peek()]:
                opStack.push(token)
            else:
                topToken=opStack.pop()
                temp1=strStack.pop()
                temp2=strStack.pop()
                tempstr="("+temp2+topToken+temp1+")"
                strStack.push(tempstr)
                opStack.push(token)
    
    while not opStack.isEmpty():
        topToken=opStack.pop()
        temp1=strStack.pop()
        temp2=strStack.pop()
        tempstr="("+temp2+topToken+temp1+")"
        strStack.push(tempstr)       
    return strStack.pop()

            
#构建解析树
def buildParseTree(fpexp):
    fpexp=" ".join(fpexp)
    fplist=fpexp.split()
    print(fplist)

    pStack=Stack()
    eTree=BinaryTree('')
    pStack.push(eTree)
    currentTree=eTree
    
    temp=0
    for i in fplist:
        if i=='(':
            currentTree.insertLeft('')
            pStack.push(currentTree)
            currentTree=currentTree.getLeftChild()
        elif i not in '+-*/)':    
            currentTree.setRootVal(eval(i))
            parent=pStack.pop()
            currentTree=parent
        elif i in '+-*/':
            currentTree.setRootVal(i)
            currentTree.insertRight('')
            pStack.push(currentTree)
            currentTree=currentTree.getRightChild()
        elif i==')':
            currentTree=pStack.pop()
        else:
            raise ValueError('Unknown Operator:'+i)
    temp+=1
    return eTree


def evaluate(parseTree):
    opers={'+':operator.add,'-':operator.sub,'*':operator.mul,'/':operator.truediv}
    leftC=parseTree.getLeftChild()
    rightC=parseTree.getRightChild()
    if leftC and rightC:
        fn=opers[parseTree.getRootVal()]
        return fn(evaluate(leftC),evaluate(rightC))
    else:
        return parseTree.getRootVal()



mylist='(4*8)/6-3'
mystr=str_repair(mylist)
myParseTree=buildParseTree(mystr)
myParseTree.printexp()
print(evaluate(myParseTree))











