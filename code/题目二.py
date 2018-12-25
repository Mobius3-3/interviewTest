#coding=utf-8
# 运行环境python2.7
import pandas as pd
import numpy as np

##
# @dev:load data from csv file and transform raw data into only containing true value of 'Label' and satisfying miniSupport >= 0.1
# @params: filename -> directory of csv file
def loadData(filename):
    
    # load data and separate header 
    data = pd.read_table(filename,header=None,sep=',')
    head = np.array(data)[0]
    data = pd.read_table(filename,header=0,sep=',')
    minSupport = 0.1 * len(data)

    # transform data into int
    for u in data.columns:
        data[u] = data[u].astype('int')
        

    dataSet = np.array(data)
    new_data = []
    row_remain = []
    head_remain = []

    n_column = np.array(data).shape[1]
    n_row = np.array(data).shape[0]

    # delete false value of 'Label'
    count = 0
    for i in range(n_row):      
        if dataSet[i-count,-1] == 0:
            dataSet = np.delete(dataSet,i-count,axis = 0)
            count = count + 1

    # choose items satisfying minSupport >= 0.1
    for j in range(n_column):
        a = np.zeros((n_column))
        for i in range(len(dataSet)):
            if dataSet[i][j] != 0:
                a = a + dataSet[i]
        
        if a[j] >= minSupport and a[-1] >= minSupport:
            new_data.append(dataSet[:,j])
            row_remain.append(j)

            head_remain.append(head[j])
            
    data = np.array(new_data).transpose()
    row  = data.shape[1]
    
    # delete the row of Label
    data = np.delete(data, row-1, 1) 

    head = np.array(head_remain)
    height = np.delete(head, row-1, 0) 
   
    n_row, n_col = data.shape

    # replace int with items' names
    a = 0
    dataSet = []
    for i in range(n_row):
        dataSet.append([])
        for j in range(n_col):
            if data[i][j] != 0:
                dataSet[a].append(height[j])
        a = a + 1
    return dataSet

#### 下面是apriori算法的相关函数

##
# @dev: create sets with single item
# @params: dataSet -> dataSet for data mining
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset, C1)

##
# @dev: for transactions in  dataset D if they are subsets of Ck and support >= minSupport, then add to retList
# @params: D -> dataset
# @params: Ck -> choose subset of Ck
# @params: minSupport -> minSupport for transactions
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                ssCnt[can] = ssCnt.get(can, 0) + 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

##
# @dev: generate Ck of different number of items
# @params: Lk -> dataset of transactions
# @params: k -> k-1 equals to number of items
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            # 前k-2项相同时，将两个集合合并
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

##
# @dev: calculate rules
# @params: dataSet -> dataset of transactions
# @params: minSupport -> minSupport of dataSets
def apriori(dataSet, minSupport=0.7):
    ## create frozen sets
    C1 = createC1(dataSet)
    ## map dataSet list to set
    D = map(set, dataSet)
    
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0): # every transactions reserve k-1 items
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

dataSet = loadData('../data/Test2_Data.csv')
L, suppDat = apriori(dataSet, 0.7)

k1=[]
k2=[]
for k,v in suppDat.iteritems():
    if v>=0.7 and len(k) == 1:
        k1.append(k)
        print(str(list(k)[0])+ ' -> Label,confidence = '+str(v))
for k,v in suppDat.iteritems():
    if v>=0.7 and len(k) == 2:
        k2.append(k)
        print(str(list(k)[0])+',  '+str(list(k)[1]) + ' -> Label,confidence = '+str(v))
print('\n总rules数、左侧1项的rules数、左侧为2项的rules数分别为：%d %d %d'%(len(k1+k2),len(k1),len(k2))).decode('UTF-8').encode('GBK')

