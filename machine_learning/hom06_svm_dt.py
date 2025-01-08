import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import log
import operator
def createDataSet():
    dataSet = [['','蜷缩','浊响','清晰','凹陷','硬滑','yes'],
               ['乌黑','蜷缩','沉闷','清晰','凹陷','','yes'],
               ['乌黑','蜷缩','','清晰','凹陷','硬滑','yes'],
               ['青绿','蜷缩','沉闷','清晰','凹陷','硬滑','yes'],
               ['','蜷缩','浊响','清晰','凹陷','硬滑','yes'],
               ['青绿','稍蜷','浊响','清晰','','软粘','yes'],
               ['乌黑','稍蜷','浊响','稍糊','稍凹','软粘','yes'],
               ['乌黑','稍蜷','浊响','','稍凹','硬滑','yes'],
               ['乌黑','','沉闷','稍糊','稍凹','硬滑','no'],
               ['青绿','硬挺','清脆','','平坦','软粘','no'],
               ['浅白','硬挺','清脆','模糊','平坦','','no'],
               ['浅白','蜷缩','浊响','模糊','平坦','软粘','no'],
               ['','稍蜷','浊响','稍糊','凹陷','硬滑','no'],
               ['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑','no'],
               ['乌黑','稍蜷','','清晰','','软粘','no'],
               ['浅白','蜷缩','浊响','模糊','平坦','硬滑','no'],
               ['青绿','','沉闷','稍糊','稍凹','硬滑','no']]
    labels=['色泽','根蒂','敲声','纹理','脐部','触感']
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
    ori_dataSet = [item for item in dataSet]
    baseEntropy = calcInfoEnt(dataSet)
    baseGainRatio = 0.0
    baseGain = 0.0
    bestFeature = -1
    numFeatures = len(dataSet[0]) - 1
    # 建立一个字典，用来存储每一个连续型特征所对应最佳切分点的具体值
    bestSplitDic = {}
    
    #print('dataSet[0]:' + str(dataSet[0]))
    for i in range(numFeatures):
        dataSet = []
        featVals = []
        # 获取第i个特征的特征值
        for t in range(len(ori_dataSet)):
            if ori_dataSet[t][i] != '':
                featVals.append(ori_dataSet[t][i])
                dataSet.append(ori_dataSet[t])
        # featVals = [example[i] for example in dataSet]
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
                try:
                    splitInfo -= prob0 * log(prob0, 2)
                except Exception:
                   print(prob0)
                splitInfo -= prob1 * log(prob1, 2)
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
        featVals = [example[bestFeature] for example in dataSet if example[bestFeature]!='']
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
    

def tree_acc(tree,dataset):
    data_sum = dataset.shape[0]
    pre_true_sum = 0
    for i in range(dataset.shape[0]):
        tree_pre_result = get_result(dataset.loc[i,], tree)
        ground_result = dataset.loc[i,][-1]
        if tree_pre_result == ground_result:
            pre_true_sum +=1
    return pre_true_sum/data_sum

def get_result(array,tree):
    if type(tree) == str:
        return tree
    else:
        key = [key for key in tree.keys()][0]
        temp_dic = [value for value in tree.values()][0]
        temp_value = array[int(key)-1]
        compare_value_list = [value for value in temp_dic]
        if temp_value > float(compare_value_list[0].replace('>',"")):
            new_tree = temp_dic[compare_value_list[0]]
        else:
            new_tree = temp_dic[compare_value_list[1]]
        result = get_result(array, new_tree)
        return result
        #new_tree = value[]

            
    
if __name__ == '__main__':
    # 统计每个特征的取值情况作为全局变量, 空值不算做一个取值
    
    
    col_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
             "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "result"]
    data = pd.read_csv("adult.data", names=col_names).head(1000)
    data_iris = pd.read_csv("iris.data",names=["1","2","3","4","result"])
    data=data.replace(" ?",np.NaN)
    data=data.dropna()
    data=data.drop_duplicates()

    #workclassList=[" Private"," Self-emp-not-inc"," Self-emp-inc"," Federal-gov"," Local-gov"," State-gov"," Without-pay"," Never-worked"]
    #educationList=[" Bachelors"," Some-college"," 11th"," HS-grad"," Prof-school"," Assoc-acdm"," Assoc-voc"," 9th"," 7th-8th"," 12th"," Masters"," 1st-4th"," 10th"," Doctorate"," 5th-6th"," Preschool"]
    #marital_status=[" Married-civ-spouse"," Divorced"," Never-married"," Separated"," Widowed"," Married-spouse-absent"," Married-AF-spouse"]
    #occupationList=[" Tech-support"," Craft-repair"," Other-service"," Sales"," Exec-managerial"," Prof-specialty"," Handlers-cleaners"," Machine-op-inspct"," Adm-clerical"," Farming-fishing"," Transport-moving"," Priv-house-serv"," Protective-serv"," Armed-Forces"]
    #relationshipList=[" Wife"," Own-child"," Husband"," Not-in-family"," Other-relative"," Unmarried"]
    #raceList=[" White",' Asian-Pac-Islander',' Amer-Indian-Eskimo',' Other',' Black']
    #native_country=" United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands".split()

    #预处理
    data.loc[data['sex']==' Female','sex']=0
    data.loc[data['sex']==' Male','sex']=1
    data.loc[data['result']==' <=50K','result']=0
    data.loc[data['result']==' >50K','result']=1

    #Dummies矩阵
    dum_workclass=pd.get_dummies(data["workclass"],prefix="workclass")
    dum_education=pd.get_dummies(data["education"],prefix="education")
    dum_occupation=pd.get_dummies(data["occupation"],prefix="occupation")
    dum_marital=pd.get_dummies(data["marital-status"],prefix="marital-status")
    dum_relationship=pd.get_dummies(data["relationship"],prefix="relationship")
    dum_race=pd.get_dummies(data["race"],prefix="race")
    dum_native=pd.get_dummies(data["native-country"],prefix="native-country")

    data1 = data.drop(["workclass",'fnlwgt',"education", "marital-status", "occupation","relationship", "race","native-country"], axis = 1)
    data2 = data1.join([dum_workclass,dum_education,dum_occupation,dum_marital,dum_relationship,dum_race,dum_native], how='outer')

    
    #训练模型
    X=data2.drop(["result"],axis=1).values.astype(float)
    Y=data2['result'].values.astype(int)
    x_iris=data_iris.drop(["result"],axis=1).values.astype(float)
    y_iris=data_iris['result']
    from sklearn import model_selection
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=1)
    x_iris_train,x_iris_test,y_iris_train,y_iris_test = model_selection.train_test_split(x_iris,y_iris,test_size=0.3,random_state=1)
    
    '''svm 线性核函数'''
    '''from sklearn import svm
    #创建分类器对象
    l_clf=svm.SVC(kernel="linear",gamma=0.6,C=1.0)
    l_clf.fit(X_train,y_train)
    print("收入预测线性核函数：",l_clf.score(X_test,y_test))
    l_clf.fit(x_iris_train,y_iris_train)
    print("iris线性核函数：",l_clf.score(x_iris_test,y_iris_test))'''

    '''svm 高斯核函数'''
    '''r_clf = svm.SVC(kernel='rbf', gamma=0.7, C = 1.0)
    r_clf.fit(X_train, y_train)
    print("收入预测高斯核函数：", r_clf.score(X_test, y_test) )
    r_clf.fit(x_iris_train,y_iris_train)
    print("iris高斯核函数：",r_clf.score(x_iris_test,y_iris_test))'''

    '''id4和c4.5'''
    from sklearn import tree
    dtc=tree.DecisionTreeClassifier(criterion="entropy",
                                    max_depth=10,
                                    max_leaf_nodes=100,
                                    min_samples_leaf=8)
    dtc.fit(X_train,y_train)
    print("收入预测id4决策树：", dtc.score(X_test, y_test) )
    dtc.fit(x_iris_train,y_iris_train)
    print("irisid4决策树：",dtc.score(x_iris_test,y_iris_test))
    
    data2_100 = data.head(100)
    list_person = np.array(data2_100).tolist()
    label_person = list(data)
    label_person.pop()
    mytree = createTree(list_person, label_person)
    print("最终构建的树为：\n",mytree)
    print("C4.5树为：",tree_acc(mytree, data_iris))
    '''list_iris = np.array(data_iris).tolist()
    label_iris = list(data_iris)
    label_iris.pop()
    mytree = createTree(list_iris, label_iris)
    print("C4.5数形状：",mytree)
    print("C4.5树为：",tree_acc(mytree, data_iris))'''

    from sklearn.neural_network import MLPClassifier
    mlp=MLPClassifier(hidden_layer_sizes=(10,),learning_rate_init=0.1,max_iter=500)
    mlp.fit(X_train,y_train)
    print("收入预测BP网络：", mlp.score(X_test, y_test) )
    mlp.fit(x_iris_train,y_iris_train)
    print("irisBP网络：",mlp.score(x_iris_test,y_iris_test))
    