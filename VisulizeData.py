
# coding: utf-8

# In[1041]:

import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pydotplus
import math
import sys
from sets import Set
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import re 
from graphviz import Digraph
from sklearn import linear_model
from __future__ import division # ensures that default division is real number division
#get_ipython().magic(u'matplotlib inline')
#%matplotlib
mpl.rc('figure', figsize=[10,6]) 


# In[1062]:

vertex = pd.read_csv('data2/VERTEX.csv')
for i in range(len(vertex)):
    vertex.iloc[i,0] = vertex.iloc[i,0]-1
edge = pd.read_csv('data2/EDGE.csv')
print vertex.shape
print edge.shape


# In[1063]:

vertex.head()
#vertex[vertex['VERTEXID']==117]


# In[1064]:

edge.head()
edge.iloc[:10,:]


# In[1065]:

dot = Digraph(comment='graph')
vertex = np.asarray(vertex)
print vertex[117][:5]
edge = np.asarray(edge)
for i in range(len(vertex)):
    dot.node(str(vertex[i][2]),str(vertex[i][0:2]))
for j in range(len(edge)):
    dot.edge(str(edge[j][3]),str(edge[j][4]))


# In[1066]:

dot.render('test-output/graph2')


# In[1067]:

NodeNum = len(vertex)
EdgeNum = len(edge)
print NodeNum
print EdgeNum
Size = (NodeNum,NodeNum+1)
AdjMatrix = np.zeros(Size)

for k in range(NodeNum):
    AdjMatrix[k][0] = k+1

for i in range(len(edge)):
#for i in range(5):
    for j in range(len(vertex)):
        if (vertex[j][2] == edge[i][3]):
            
            startV = vertex[j][0]
            #print "start"
            #print startV
        if(vertex[j][2] == edge[i][4]):
            #print "end"
            endV = vertex[j][0]
            #print endV
    AdjMatrix[endV][startV+1] = 1
#print AdjMatrix
print AdjMatrix[:10,:]


# In[1068]:

def updateFlag(val,sorted_attr_row):
    for i in range(len(sorted_attr_row)-1):
        val[i] = sorted_attr_row[i+1]

def defineT(attr):
    dtype = [('Id',int)]
    for i in range(len(attr)):
        dtype.append((attr[i],'S10'))
    return dtype


# In[1069]:

def ACompatible(graph,edge,attr):
    dtype = defineT(attr)
    value = list(graph[attr].itertuples())
    original_table = np.asarray(value,dtype=dtype)
    groupArr = []

    sorted_attr = np.sort(original_table, order=attr)  
    val = ['','']
    for i in range(len(sorted_attr)):
        for j in range(len(attr)):
            if(sorted_attr[i][j+1]==val[j]):
                if(j == len(attr)-1):
                    newGroup.append(sorted_attr[i])
            else:
                newGroup = [(sorted_attr[i])]
                groupArr.append(newGroup)
                updateFlag(val,sorted_attr[i])
                break
    return groupArr


# In[1070]:

def DataStruture(result1,edge,attr,graph):
    print "update data structure"
    print "groupnumber",len(result1)
    groupNum = len(result1)
    nodeNum = graph.shape[0]
    mapsize = (nodeNum,groupNum)
    bitMap = np.zeros(mapsize)
    PArraySize = (groupNum,groupNum)
    PArray = np.zeros(PArraySize)
    edge = np.asarray(edge)
    
    #initialize bit map
    for i in range(groupNum):
        groupSet = result1[i]
        for j in range(len(groupSet)):
            CurrentNode = groupSet[j]
            #print CurrentNode
            index = CurrentNode[0]
            for m in range(groupNum):
                if(m!=i):
                    groupSet1 = result1[m]
                    for n in range(len(groupSet1)):
                        CurrentNode2 = groupSet1[n]
                        #print CurrentNode
                        index2 = CurrentNode2[0]
                        if(not pd.isnull(edge[index2,index+1])):
                            if(edge[index2,index+1]==1):
                                bitMap[index][m]=1
            

    #print "bitmap length",len(bitMap)
    #print bitMap[117],bitMap[125]

    #initialize participation array
    for i in range(groupNum):
        groupSet = result1[i]
        temp = np.zeros(groupNum)
        for j in range(len(groupSet)):
            CurrentNode = groupSet[j]
            index = CurrentNode[0]
            for k in range(groupNum):
                temp[k] += bitMap[index][k]
        PArray[i] = temp

    #print "parray"
    #print PArray
    return PArray,bitMap


# In[1071]:

def condition(SubGroup,PArray):
    GroupNum = len(SubGroup);
    for i in range(GroupNum):
        setSize = len(SubGroup[i])
        for j in range(GroupNum):
            if ((PArray[i][j]!=setSize) & (PArray[i][j]!=0.0)):
                return False,i+1
    return True,0
                


# In[1072]:

def Split(BitMap,fixedGNum,TempResult,graph,attr):
    groupNum = len(TempResult)
    WaitingGroup = TempResult[fixedGNum-1]
    #print "waitinggroup"
    #print WaitingGroup
    tempBitMapsize = (len(WaitingGroup),groupNum)
    tempBitMap = np.zeros(tempBitMapsize)
    NodeIndex = []
    subGroup1 = []
    subGroup2 = []
    
    dtype = defineT(attr)
    value = list(graph[attr].itertuples())
    original_table = np.asarray(value,dtype=dtype)
    #print original_table
    
    for i in range(len(WaitingGroup)):
        NodeIndex.append(WaitingGroup[i][0])
        tempBitMap[i][:] = BitMap[WaitingGroup[i][0]][:]
    #print "nodeindex"
    #print NodeIndex
    #print "temp bit map"
    #print tempBitMap
    #print "node number",len(np.asarray(NodeIndex))
    #print "bit size",len(tempBitMap)
    table = np.concatenate((np.asarray(NodeIndex).reshape(len(np.asarray(NodeIndex)),1),tempBitMap),axis=1)

    typename = []
    for i in range(groupNum):
        typename.append("attr"+str(i))
    
    dtype = [('Id',int)]
    for i in range(groupNum):
        dtype.append((typename[i],float))
    #print dtype

    #print table
    value = []
    for i in range(len(NodeIndex)):
        value.append(tuple(table[i].tolist()))

    waitingSortTable = np.asarray(value,dtype=dtype)
    #print table
    
    for k in range(groupNum):
        temp = np.sort(waitingSortTable, order=[typename[k],'Id']) 
        #print "temp sort attribute"
        #print temp
        for m in range(len(temp)-1):
            if(temp[m][k+1]!=temp[m+1][k+1]):
                #print "found it"
                #print m
                for j in range(m+1):
                    subGroup1.append(original_table[temp[j][0]])
                for n in range(m+1,len(temp)):
                    subGroup2.append(original_table[temp[n][0]])
                #print "group1"
                #print subGroup1
                #print "group2"
                #print subGroup2
                #print TempResult
                TempResult.remove(WaitingGroup)
                TempResult.append(subGroup1)
                TempResult.append(subGroup2)
                #print "result",TempResult
                return TempResult
    return none
        


# In[1073]:

def SNAP(graph,edge,attr):
    TempResult = ACompatible(graph,edge,attr)
    PArray,BitMap = DataStruture(TempResult,edge,attr,graph)
    #for i in range(len(TempResult)):
    #    print "another group"
    #    print TempResult[i]
    cond,groupNum = condition(TempResult,PArray)
    while(not cond):
    #for j in range(5):
        TempResult = Split(BitMap,groupNum,TempResult,graph,attr)
        PArray,BitMap = DataStruture(TempResult,edge,attr,graph)
        cond,groupNum = condition(TempResult,PArray)
    for i in range(len(TempResult)):
        print "another group"
        print TempResult[i]
    print PArray
    return TempResult,PArray


# In[1075]:

attr = ['TYPE']
vertex = pd.read_csv('data2/VERTEX.csv')
for i in range(len(vertex)):
    vertex.iloc[i,0] = vertex.iloc[i,0]-1

SMRnode,PArray = SNAP(vertex,AdjMatrix,attr)


# In[1076]:

dot1 = Digraph(comment='Summary Graph2')
#vertex = np.asarray(vertex)
#edge = np.asarray(edge)
#for i in range(len(vertex)):
#    dot.node(str(vertex[i][2]),str(vertex[i][0]))
#for j in range(len(edge)):
#    dot.edge(str(edge[j][3]),str(edge[j][4]))
    
print AdjMatrix[SMRnode[1][0][0],SMRnode[1][0][0]]

for i in range(len(SMRnode)):
    dot1.node(str(SMRnode[i][0]),str(SMRnode[i][0]))
    for j in range(len(SMRnode)):
        if(i!=j):
            if(PArray[i][j]!=0):
                for k in range(int(PArray[i][j])):
                    dot1.edge(str(SMRnode[i][0]),str(SMRnode[j][0]))

dot1.render('test-output/summary_Graph2')


# In[1077]:

dot2 = Digraph(comment='ColorGraph2')
vertex = pd.read_csv('data2/VERTEX.csv')
#vertex = np.asarray(vertex)
edge = np.asarray(edge)
for i in range(len(SMRnode)):
    if(i==0):
        color = 'red'
    if(i==1):
        color = 'blue'
    if(i==2):
        color = 'green'
    if(i==3):
        color = 'yellow'
    if(i==4):
        color = 'cyan'
    if(i==5):
        color = 'magenta'
    if(i==6):
        color = 'Purple'
    if(i==7):
        color = 'grey'
    if(i==8):
        color = 'tan'
    if(i==9):
        color = 'MidnightBlue'
    if(i==10):
        color = 'chocolate'
        
    for j in range(len(SMRnode[i])):
        #print SMRnode[i][j]
        hashvalue = vertex.iloc[SMRnode[i][j][0]]['HASH']
        dot2.node(str(hashvalue),str(SMRnode[i][j]),color = color,style='filled')
for j in range(len(edge)):
    dot2.edge(str(edge[j][3]),str(edge[j][4]))
dot2.render('test-output/ColorGrap2')


# In[1026]:

vertex = pd.read_csv('data1/VERTEX.csv')
print vertex.head()
t = vertex[vertex['VERTEXID']==1]['HASH']
print t
print vertex.iloc[0]['HASH']


# In[ ]:



