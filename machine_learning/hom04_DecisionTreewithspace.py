import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


#计算信息熵
def cal_information_entropy(data):
    data_label = data.iloc[:,-1] #类别标签
    label_class = data_label.value_counts() #总共有多少类
    Ent = 0
    D_W_x = sum(data['weights']) #整体数据的权值和
    for k in label_class.keys():
        D_k_wx = sum(data.loc[data_label == k]['weights']) #每个类别的权值和
        p_k = D_k_wx / D_W_x
        Ent += -p_k*np.log2(p_k)
    return Ent

#计算信息增益
#计算给定数据属性a的信息增益
def cal_information_gain(data, a, p): #p表示课本中的ρ，即非空数据占整体数据样本的比例
    Ent = cal_information_entropy(data) #整体信息熵
    feature_class = data[a].value_counts() #特征有多少种可能
    gain = 0
    for v in feature_class.keys():
        r = sum(data[data[a] == v]['weights'])/sum(data['weights']) # 课本上的r，表示该特征某一个取值的样本权值和占整体样本权值和的比值
        Ent_v = cal_information_entropy(data.loc[data[a] == v])
        gain += r*Ent_v
    return p*(Ent - gain)

#获取标签最多的那一类
def get_most_label(data):
    data_label = data.iloc[:,-1]
    label_sort = data_label.value_counts(sort=True)
    return label_sort.keys()[0]

#挑选最优特征，即信息增益最大的特征
def get_best_feature(data):
    features = data.columns[1:-1]
    res = {}
    for a in features:
        data_not_null = data.dropna(axis=0,subset = [a]) #该特征不为空的数据
        p = sum(data_not_null['weights']) / sum(data['weights']) #占比
        temp = cal_information_gain(data_not_null, a, p ) #用非空的数据去算信息增益,最后乘上p
        res[a] = temp
    res = sorted(res.items(),key=lambda x:x[1],reverse=True) #按照信息增益排名
    return res[0][0]

##将数据转化为（属性值：数据）的元组形式返回，并删除之前的特征列
def drop_exist_feature(data, best_feature):
    attr = pd.unique(data.dropna(axis=0,subset = [best_feature])[best_feature]) #最佳划分特征的取值可能,先不包括空值
    res = []
    data_non = data[data[best_feature].isna()] #该特征为空的数据
    for val in attr:
        new_data = data[data[best_feature] == val]
        p = len(new_data) / len(data) #计算当前取值占比
        if len(data_non) > 0: #如果有的话
            data_non_cp = data_non.copy()
            data_non_cp['weights'] *= p #权值变小
            new_data = new_data.append(data_non_cp) #并入数据
        res.append((val, new_data))
    final_data = [(n[0], n[1].drop([best_feature], axis=1)) for n in res] #删除用过的特征
    return final_data

#创建决策树
def create_tree(data):
    data_label = data.iloc[:,-1]
    if len(data_label.value_counts()) == 1: #只有一类
        return data_label.values[0]
    if all(len(data[i].value_counts()) == 1 for i in data.iloc[:,:-1].columns): #所有数据的特征值一样，选样本最多的类作为分类结果
        return get_most_label(data)
    best_feature = get_best_feature(data) #根据信息增益得到的最优划分特征
    Tree = {best_feature:{}} #用字典形式存储决策树
    exist_vals = pd.unique(data[best_feature])  # 当前数据下最佳特征的取值
    if len(exist_vals) != len(column_count[best_feature]):  # 如果特征的取值相比于原来的少了
        no_exist_attr = set(column_count[best_feature]) - set(exist_vals)  # 少的那些特征
        for no_feat in no_exist_attr:
            Tree[best_feature][no_feat] = get_most_label(data)  # 缺失的特征分类为当前类别最多的
    for item in drop_exist_feature(data,best_feature): #根据特征值的不同递归创建决策树
        Tree[best_feature][item[0]] = create_tree(item[1])
    return Tree

def predict(Tree , test_data):
    first_feature = list(Tree.keys())[0]
    second_dict = Tree[first_feature]
    input_first = test_data.get(first_feature)
    input_value = second_dict[input_first]
    if isinstance(input_value , dict): #判断分支还是不是字典
        class_label = predict(input_value, test_data)
    else:
        class_label = input_value
    return class_label
# 定义文本框和箭头格式
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.serif'] = ['SimHei']
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
    data = pd.read_csv('wm2_0a.csv')
    # 统计每个特征的取值情况作为全局变量, 空值不算做一个取值
    column_count = dict([(ds, list(pd.unique(data.dropna(axis=0,subset = [ds])[ds]))) for ds in data.iloc[:, :-1].columns])
    data.insert(0, 'weights', 1) #插入每个样本权值
    tree = create_tree(data)
    print(tree)
    createPlot(tree)
