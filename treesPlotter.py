# coding:utf-8
import matplotlib.pyplot as plt
import matplotlib

myfont = matplotlib.font_manager.FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 获取树的宽度
def getNumLeaves(myTree):
    numLeaves = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]

    # 检查宽度
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeaves += getNumLeaves(secondDict[key])
        else:
            numLeaves += 1
    return numLeaves


# 获取树的深度
def get_tree_depth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]

    # 检查深度
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + get_tree_depth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 只是在抄书上的代码，不怎么能理解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction', va='center',
                            ha='center', bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.axl = plt.subplot(111, frameon=False)
    plotNode(u"决策节点", (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode(u"叶节点", (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


createPlot()
