from copy import deepcopy

from numpy import *
import numpy as np
import operator
import csv
import random
import math


# 计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt


# 将数据分为featName 以及 带标准化的label的数据
def dataNormalized(dataSet):
    labels = dataSet[0]
    # print("FeatNames:")
    # print(labels)
    featLength = len(dataSet[0]) - 1
    dataVial = dataSet[1:]
    dataVial = np.array(dataVial).astype(float)

    # 使用np库计算每列的最大最小值
    max1 = np.max(dataVial, 0)
    min1 = np.min(dataVial, 0)
    print(max1)
    print(min1)
    # 对每个数据做归一化
    for example in dataVial:
        for n in range(0, featLength):
            example[n] = (example[n] - min1[n]) / (max1[n] - min1[n])
    dataVial = dataVial.tolist()
    print("dataNormalized:")
    print(dataVial)
    return dataVial, labels


# 决策树需要对数据进行连续值处理
# 此处代码专为winequality数据集准备
def dataDispersed(dataSet):
    labels = dataSet[0]
    featLength = len(dataSet[0]) - 1
    dataVial = dataSet[1:]
    dataVial = np.array(dataVial).astype(float)
    # 使用np库计算每列的最大最小值
    max1 = np.max(dataVial, 0)
    min1 = np.min(dataVial, 0)
    # 为节约速度 牺牲空间先求出1/10差值
    delta = []
    for n in range(0, featLength):
        delta.append((max1[n] - min1[n]) / 10)
    # 将数据每个划分为1-10个部分
    for example in dataVial:
        for n in range(0, featLength):
            example[n] = int(((example[n] - min1[n]) // delta[n]) + 1)
    print("dataDispersed:")
    print(dataVial)
    return dataVial, labels


# 使用留出法将原数据集划分为 8:2 的数据集合测试集
# 修改留出法，没有遵循分层采样、互斥采样的原则，所以做重新修改
# 将数据集分层拆分为5个独立的集，并用于计算、评估
# important
def dataSplitToFive(dataSet):
    # 获取所有的label值
    featList = [rec[-1] for rec in dataSet]
    # 创建无重复label集合
    uniqueVials = set(featList)
    originSet = [[], [], [], [], [], [], [], [], [], []]

    # 一个看起来就非常慢的双循环将所有数据按label分到自己的类别中
    # 拿一个numRec记录每个分类有多少个
    for label1 in uniqueVials:
        label1 = int(label1)
        for rec1 in dataSet:
            if rec1[-1] == label1:
                originSet[label1].append(rec1)

    # 做完了输出一哈结果
    print("收集到的label如下：")
    print(uniqueVials)
    print("每个label的数量统计如下：")
    for rec1 in originSet:
        print(len(rec1))

    count = np.zeros((10, 1))
    splitSets = [[], [], [], [], []]
    for labelName in uniqueVials:
        labelName = int(labelName)
        count[labelName] = math.ceil(len(originSet[labelName]) / 5)

    print(count)
    # 有五个数据集，每个数据集对原集合中所有的label分别采样指定次数，并解决无法整除的问题
    for n in range(5):
        for labelName in uniqueVials:
            labelName = int(labelName)
            num = deepcopy(count[labelName])
            while originSet[labelName] and num > 0:
                temp = (originSet[labelName].pop(random.randint(0, len(originSet[labelName]) - 1))).tolist()
                splitSets[n].append(temp)
                num -= 1

    # 输出划分结果：
    print("划分数据集完成！")
    for n in range(5):
        print(len(splitSets[n]))
    return splitSets


# 创建数据集
def createDataSet():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


# 划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 取到下标为axis的特征进行比对，并取出下标前的数据、下标后的数据extend之后作为一个列表append到dataSet里面
            reducedFeatVec = featVec[:axis]
            # extend 合并列表元素 append 将列表作为一个新元素插入
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 对数据列表统计每个分类的数量,并返回数量最多的特征名 工具类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 遍历最好数据划分 返回最佳特征下标int
# important
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 获取目标特征的所有属性
        featList2 = [example[i] for example in dataSet]
        # 创建无重复属性集合
        uniqueVials1 = set(featList2)
        newEntropy = 0.0
        # 遍历每个属性，计算单个属性划分之后的信息增益大小
        # 累加所有属性的加权信息增益，计算增益率
        for value in uniqueVials1:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # 如果信息增益率最大，则存储该特征为划分值
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 递归构造dict形式的决策树 dataSet是有label的list，featNames是记录特征名称的list
# important
def createTree(dataSet, featName):
    # 判断是否达到递归的特殊条件
    featNames = featName[:]
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优划分特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = featNames[bestFeat]
    # 预定义决策字典，从待选特征中删除已选特征
    myTree = {bestFeatLabel: {}}
    del (featNames[bestFeat])
    # 从数据集中获取特征的属性种类uniqueVials
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVials = set(featValues)
    # 按照特征属性的不同将数据集进行划分,并输入到字典中
    for value in uniqueVials:
        subLabel = featNames[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabel)
    return myTree


# 使用决策树对测试数据集进行分类
def classify(inputTree, featLabel, testRec):
    featLabels = featLabel[:]
    # 获取当前字典的特征名
    feat = list(inputTree.keys())[0]
    # 使用这个特征名从决策树中获取该特征分类下的所有属性分类字典
    # print(featLabels)
    nextDict = inputTree[feat]
    # 获取这个分类在feat中的下标
    featIndex = featLabels.index(feat)
    # 对于该属性分类字典中的每一个属性，如果测试值中的该属性值与该属性相同，则进行判定：
    # 如果该属性分类字典中依旧还有字典，说明决策树还没有到底，继续递归求结果。
    # 如果该属性分类字典中已经没有字典了，说明决策树到底，返回其结果。
    for featValue in nextDict.keys():
        if testRec[featIndex] == featValue:
            if type(nextDict[featValue]).__name__ == 'dict':
                classLabel = classify(nextDict[featValue], featLabels, testRec)
            else:
                classLabel = nextDict[featValue]
                # print("rec:")
                # print(classLabel)
            return classLabel


# 对批量数据进行分类
def batchClassifyAndEstimate(inputTree, featLabels, testSet1):
    total = len(testSet1)
    count = 0
    for rec1 in testSet1:
        ansLabel = classify(inputTree, featLabels, rec1)
        if ansLabel == rec1[-1]:
            count += 1
    correctRating = (count / total) * 100
    print("使用 2 8 留出法测试正确率")
    print(correctRating, "%")
    return correctRating


with open("bill_authentication.csv") as wineInfo:
    csvReader = csv.reader(wineInfo)
    wineInfoList = list(csvReader)
    # 拆解数据 和 标签名
    data, label = dataDispersed(wineInfoList)
    splitDatas = dataSplitToFive(data)
    # print(splitData)
    # data, label = createDataSet()
    data = data.tolist()

    for n in range(5):
        splitData = splitDatas[:]
        testSet = splitData.pop(n)
        createSet = []
        for set1 in splitData:
            createSet.extend(set1)
        tree = createTree(createSet, label)
        batchClassifyAndEstimate(tree, label, testSet)

