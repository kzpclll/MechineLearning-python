from numpy import *
import numpy as np
import operator
import csv
import random


# 数据处理与公用接口 <--                       -->
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
def dataSplitToTestData(dataSet):
    length = int(len(dataSet) / 5)
    testSet = []
    for i in range(length):
        testSet.append(dataSet.pop(random.randint(0, len(dataSet) - 1)))
    return dataSet, testSet


# 创建数据集
def createDataSet():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


# 划分数据集 将数据集中的下标为axis的数据剔除
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
    sortedClassCount = sorted(classCount.iteritemsd(), key=operator.itemgetter(1), reverse=True)
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
        featList = [example[i] for example in dataSet]
        # 创建无重复属性集合
        uniqueVials = set(featList)
        newEntropy = 0.0
        # 遍历每个属性，计算单个属性划分之后的信息增益大小
        # 累加所有属性的加权信息增益，计算增益率
        for value in uniqueVials:
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
def createTree(dataSet, featNames):
    # 判断是否达到递归的特殊条件
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
# ready to do : understand this ->
def classify(inputTree, featLabels, testRec):
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
                print("rec:")
                print(classLabel)
            return classLabel


# 对批量数据进行分类
def batchClassify(inputTree, featLabels, testSet):
    for rec in testSet:
        classify(inputTree, featLabels, rec)




with open("winequality-red.csv") as wineInfo:
    csvReader = csv.reader(wineInfo)
    wineInfoList = list(csvReader)
    data, label = dataDispersed(wineInfoList)
    # data, label = createDataSet()
    data = data.tolist()
    data, testData = dataSplitToTestData(data)
    labelCreate = label[:]
    tree = createTree(data, labelCreate)

    print("dict tree:")
    print(tree)
    # print("label:")
    # print(label)
    rec = [3.0, 11.0, 1.0, 1.0, 3.0, 1.0, 1.0, 4.0, 6.0, 1.0, 4.0, 3.0]
    classify(tree, label, rec)

    # batchClassify(tree, label, testData)
