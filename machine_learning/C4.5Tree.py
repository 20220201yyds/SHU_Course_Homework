from math import log
import operator
import numpy as np
#绘制决策树
import matplotlib.pyplot as plt

def createDataSet():
    '''dataSet = [['no', 'single', 125, 'no'],
               ['no', 'married', 100, 'no'],
               ['no', 'single', 70, 'no'],
               ['yes', 'married', 120, 'no'],
               ['no', 'divorced', 95, 'yes'],
               ['no', 'married', 60, 'no'],
               ['yes', 'divorced', 220, 'no'],
               ['no', 'single', 85, 'yes'],
               ['no', 'married', 75, 'no'],
               ['no', 'single', 90, 'yes']]
    labels = ['own_house', 'marriage', 'income(k)']  '''
    dataSet = [['青绿','蜷缩','浊响','清晰','凹陷','硬滑',0.697,0.46,'yes'],
               ['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑',0.774,0.376,'yes'],
               ['乌黑','蜷缩','浊响','清晰','凹陷','硬滑',0.634,0.264,'yes'],
               ['青绿','蜷缩','沉闷','清晰','凹陷','硬滑',0.608,0.318,'yes'],
               ['浅白','蜷缩','浊响','清晰','凹陷','硬滑',0.556,0.215,'yes'],
               ['青绿','稍蜷','浊响','清晰','稍凹','软粘',0.403,0.237,'yes'],
               ['乌黑','稍蜷','浊响','稍糊','稍凹','软粘',0.481,0.149,'yes'],
               ['乌黑','稍蜷','浊响','清晰','稍凹','硬滑',0.437,0.211,'yes'],
               ['乌黑','稍蜷','沉闷','稍糊','稍凹','硬滑',0.666,0.091,'no'],
               ['青绿','硬挺','清脆','清晰','平坦','软粘',0.243,0.267,'no'],
               ['浅白','硬挺','清脆','模糊','平坦','硬滑',0.245,0.057,'no'],
               ['浅白','蜷缩','浊响','模糊','平坦','软粘',0.343,0.099,'no'],
               ['青绿','稍蜷','浊响','稍糊','凹陷','硬滑',0.639,0.161,'no'],
               ['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑',0.657,0.198,'no'],
               ['乌黑','稍蜷','浊响','清晰','稍凹','软粘',0.360,0.37,'no'],
               ['浅白','蜷缩','浊响','模糊','平坦','硬滑',0.593,0.042,'no'],
               ['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑',0.719,0.103,'no']]
    labels=['色泽','根蒂','敲声','纹理','脐部','触感','密度','含糖率']
    return dataSet, labels

def calcInfoEnt(dataSet):
    """计算信息熵"""
    numEntries = len(dataSet)
    labelCounts ={}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] =0
        labelCounts[currentLabel] += 1
    Ent = 0.0
    for label in labelCounts.keys():
        prob = float(labelCounts[label])/numEntries
        Ent -= prob*log(prob,2)
    return Ent

def majorityCnt(classList):
    """取最多结果"""
    classCount = {}
    # classList= np.mat(classList).flatten().A.tolist()[0]  # 数据为[['否'], ['是'], ['是']], 转换后为['否', '是', '是']
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def splitDataSet(dataSet,axis,value):
    """对离散型特征划分数据集"""
    retDataSet = []  # 创建新的list对象，作为返回的数据
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])  # 抽取
            retDataSet.append(reducedFeatVec)
            #print("retDataSet",retDataSet)
    return retDataSet


def splitContinuousDataSet(dataSet, axis, value, direction):
    """对连续型特征划分数据集"""
    subDataSet = []
    for featVec in dataSet:
        if direction == 0:
            if featVec[axis] > value:  # 按照大于(>)该值进行划分
                reduceData = featVec[:axis]
                reduceData.extend(featVec[axis + 1:])
                subDataSet.append(reduceData)
        if direction == 1:
            if featVec[axis] <= value:  # 按照小于等于(<=)该值进行划分
                reduceData = featVec[:axis]
                reduceData.extend(featVec[axis + 1:])
                subDataSet.append(reduceData)
    return subDataSet

def chooseBestFeatureToSplit(dataSet, labels):
    """选择最好的数据集划分方式"""
    baseEntropy = calcInfoEnt(dataSet)
    baseGainRatio = 0.0
    baseGain = 0.0
    bestFeature = -1
    numFeatures = len(dataSet[0]) - 1
    # 建立一个字典，用来存储每一个连续型特征所对应最佳切分点的具体值
    bestSplitDic = {}
    #print('dataSet[0]:' + str(dataSet[0]))
    for i in range(numFeatures):
        # 获取第i个特征的特征值
        featVals = [example[i] for example in dataSet]
        # 如果该特征时连续型数据
        if type(featVals[0]).__name__ == 'float' or type(
                featVals[0]).__name__ == 'int':
            # 将该特征的所有值按从小到大顺序排序
            sortedFeatVals = sorted(featVals)
            # 取相邻两样本值的平均数做划分点，共有 len(featVals)-1 个
            splitList = []
            for j in range(len(featVals) - 1):
                splitList.append(
                    (sortedFeatVals[j] + sortedFeatVals[j + 1]) / 2.0)
            # 遍历每一个切分点
            for j in range(len(splitList)):
                # 计算该划分方式的条件信息熵newEntropy
                newEntropy = 0.0
                value = splitList[j]
                # 将数据集划分为两个子集
                greaterSubDataSet = splitContinuousDataSet(dataSet, i, value, 0)
                smallSubDataSet = splitContinuousDataSet(dataSet, i, value, 1)
                prob0 = len(greaterSubDataSet) / float(len(dataSet))
                newEntropy += prob0 * calcInfoEnt(greaterSubDataSet)
                prob1 = len(smallSubDataSet) / float(len(dataSet))
                newEntropy += prob1 * calcInfoEnt(smallSubDataSet)
                # 计算该划分方式的分裂信息
                splitInfo = 0.0
                splitInfo -= prob0 * log(prob0, 2)
                splitInfo -= prob1 * log(prob1, 2)
                # 计算信息增益率 = 信息增益 / 该划分方式的分裂信息
                '''gainRatio = float(baseEntropy - newEntropy) / splitInfo
                #print(gainRatio)
                if gainRatio > baseGainRatio:
                    baseGainRatio = gainRatio
                    bestSplit = j
                    bestFeature = i
                    bestSplitDic[labels[i]] = splitList[bestSplit]  # 最佳切分点'''
                gain = float(baseEntropy-newEntropy)
                if gain>baseGain:
                    bestGain = gain
                    bestSplit = j
                    bestFeature = i
                    bestSplitDic[labels[i]] = splitList[bestSplit]  # 最佳切分点
        else:  # 如果该特征时离散型数据
            uniqueVals = set(featVals)
            splitInfo = 0.0
            # 计算每种划分方式的条件信息熵newEntropy
            newEntropy = 0.0
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))
                splitInfo -= prob * log(prob, 2)  # 计算分裂信息
                newEntropy += prob * calcInfoEnt(subDataSet)  # 计算条件信息熵
            # 若该特征的特征值都相同，说明信息增益和分裂信息都为0，则跳过该特征
            if splitInfo == 0.0:
                continue
            # 计算信息增益率 = 信息增益 / 该划分方式的分裂信息
            '''gainRatio = float(baseEntropy - newEntropy) / splitInfo
            if gainRatio > baseGainRatio:
                bestFeature = i
                baseGainRatio = gainRatio'''
            gain = float(baseEntropy-newEntropy)
            if gain>baseGain:
                bestFeature = i
                baseGain = gain
    # 如果最佳切分特征是连续型，则最佳切分点为具体的切分值
    if type(dataSet[0][bestFeature]).__name__ == 'float' or type(
            dataSet[0][bestFeature]).__name__ == 'int':
        bestFeatValue = bestSplitDic[labels[bestFeature]]
    # 如果最佳切分特征时离散型，则最佳切分点为 切分特征名称,【其实对于离散型特征这个值没有用】
    if type(dataSet[0][bestFeature]).__name__ == 'str':
        bestFeatValue = labels[bestFeature]
    # print('bestFeature:' + str(labels[bestFeature]) + ', bestFeatValue:' + str(bestFeatValue))
    return bestFeature, bestFeatValue


def createTree(dataSet, labels):
    """创建C4.5树"""
    classList = [example[-1] for example in dataSet]
    #print(classList)
    # 如果类别完全相同，则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:
       return majorityCnt(classList)
    bestFeature, bestFeatValue = chooseBestFeatureToSplit(dataSet, labels)
    if bestFeature == -1:  # 如果无法选出最优分类特征，返回出现次数最多的类别
       return majorityCnt(classList)
    bestFeatLabel = labels[bestFeature]
    myTree = {bestFeatLabel: {}}
    subLabels = labels[:bestFeature]
    subLabels.extend(labels[bestFeature + 1:])
    # 针对最佳切分特征是离散型
    if type(dataSet[0][bestFeature]).__name__ == 'str':
        featVals = [example[bestFeature] for example in dataSet]
        uniqueVals = set(featVals)
        for value in uniqueVals:
            reduceDataSet = splitDataSet(dataSet, bestFeature, value)
            # print('reduceDataSet:' + str(reduceDataSet))
            myTree[bestFeatLabel][value] = createTree(reduceDataSet, subLabels)
            # print(myTree[bestFeatLabel][value])
    # 针对最佳切分特征是连续型
    if type(dataSet[0][bestFeature]).__name__ == 'int' or type(
            dataSet[0][bestFeature]).__name__ == 'float':
        # 将数据集划分为两个子集，针对每个子集分别建树
        value = bestFeatValue
        greaterSubDataSet = splitContinuousDataSet(dataSet, bestFeature, value, 0)
        smallSubDataSet = splitContinuousDataSet(dataSet, bestFeature, value, 1)
        #  print('greaterDataset:' + str(greaterSubDataSet))
        # print('smallerDataSet:' + str(smallSubDataSet))
        # 针对连续型特征，在生成决策的模块，修改划分点的标签，如“> x.xxx”，"<= x.xxx"
        myTree[bestFeatLabel]['>' + str(value)] = createTree(greaterSubDataSet,subLabels)
        myTree[bestFeatLabel]['<=' + str(value)] = createTree(smallSubDataSet,subLabels)
    return myTree


 


# 定义文本框和箭头格式
decisionNode = dict(boxstyle="round4", color='#3366FF')  #定义判断结点形态
leafNode = dict(boxstyle="circle", color='#FF6633')  #定义叶结点形态
arrow_args = dict(arrowstyle="<-", color='g')  #定义箭头

#绘制带箭头的注释
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


#计算叶结点数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


#计算树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


#在父子结点间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)  #在父子结点间填充文本信息
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  #绘制带箭头的注释
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD



def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
    
if __name__ == '__main__':
    plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签
    plt.rcParams['axes.unicode_minus']=False
    dataSet, labels = createDataSet()
    mytree = createTree(dataSet, labels)
    print("最终构建的C4.5分类树为：\n",mytree)
    createPlot(mytree)


