import os
import time
import numpy as np
import pandas as pd
import csv
import math
import random
import xlrd


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


# load files
FinalAllDisease = []
ReadMyCsv(FinalAllDisease, "side_effect.csv")
# ReadMyCsv(FinalAllDisease, "disease.csv")
FinalAllDisease = np.array(FinalAllDisease)[:, 0]
print(len(FinalAllDisease))
print(FinalAllDisease[1])
# FinalAllDisease = FinalAllDisease[0:100]


DiseaseMeSHTreeStructure = []
# ReadMyCsv(DiseaseMeSHTreeStructure, "MeSHTreeStructureLow.csv")
# print(len(DiseaseMeSHTreeStructure))
# print(DiseaseMeSHTreeStructure[0])

data1 = xlrd.open_workbook('ADR_Drug.xlsx')
table1 = data1.sheet_by_name('Sheet 1')
for i in range(table1.nrows):
    x = table1.cell(i, 3).value
    y = table1.cell(i, 2).value
    y = y.split('.')
    if len(y) == 4:
        y1 = 'a'+y[0]
        y2 = '0'+y[1]
        y3 = '0'+y[2]
        y = y1+'.' + y2 + '.' + y3 + '.' + y[3]
    if len(y) == 3:
        y1 = 'a' + y[0]
        y2 = '0' + y[1]
        y3 = '0' + y[2]
        y = y1 + '.' + y2 + '.' + y3
    if len(y) == 2:
        y1 = 'a' + y[0]
        y2 = '0' + y[1]
        y = y1 + '.' + y2
    if len(y) == 1:
        y = 'a' + y[0]
    DiseaseMeSHTreeStructure.append([x, y])


AllDisease = FinalAllDisease
mesh = DiseaseMeSHTreeStructure



DiseaseAndMeshID = []
counter1 = 0
while counter1 < len(AllDisease):
    DiseaseAndMeshPair = []
    DiseaseAndMeshID.append(DiseaseAndMeshPair)
    DiseaseAndMeshID[counter1].append(AllDisease[counter1])
    counter2 = 0
    flag = 0
    while counter2 < len(mesh):
        if (mesh[counter2][0] == DiseaseAndMeshID[counter1][0]) & (flag == 1):
            DiseaseAndMeshID[counter1][1].append(mesh[counter2][1])
        if (mesh[counter2][0] == DiseaseAndMeshID[counter1][0]) & (flag == 0):
            MeshID = []
            MeshID.append(mesh[counter2][1])
            DiseaseAndMeshID[counter1].append(MeshID)
            flag = 1
        if (counter2 == len(mesh) - 1) & (len(DiseaseAndMeshID[counter1]) == 1):
            DiseaseAndMeshID[counter1].append(0)
        counter2 = counter2 + 1
    counter1 = counter1 + 1
print('DiseaseAndMeshID')
print(len(DiseaseAndMeshID))
StorFile(DiseaseAndMeshID, 'DiseaseAndMeshID2.csv')



DAGs = []
counter1 = 0
while counter1 < len(AllDisease):
    group = []
    group.extend(DiseaseAndMeshID[counter1])
    group.append(0)
    group1 = []
    group1.append(group)
    DAGs.append(group1)
    counter1 = counter1 + 1
print('len(DAGs)', len(DAGs))
StorFile(DAGs, 'DAGsLeaf2.csv')



counter = 0
while counter < len(DAGs):
    print(DAGs[counter][0][1])
    print(len(DAGs[counter]))

    if DAGs[counter][0][1] == 0:
        counter = counter + 1
        continue
    counter1 = 0
    while counter1 < len(DAGs[counter]):  #################
        counter2 = 0
        while counter2 < len(DAGs[counter][counter1][1]):
            layer = DAGs[counter][counter1][2]  #######################
            # if len(DAGs[0][counter1][1][counter2]) <= 3:
            #     break
            print(len(DAGs[counter][counter1][1]))
            print(layer)
            print(len(DAGs[counter][counter1][1][counter2]))
            if len(DAGs[counter][counter1][1][counter2]) > 3:  ####################
                NID = DAGs[counter][counter1][1][counter2]  #####################
                print(NID)
                L = len(NID)
                NID = NID[0:L - 4]
                print(NID)
                counter3 = 0
                flag = 1
                while counter3 < len(mesh):
                    print(len(mesh))
                    print(mesh[counter3][1])
                    print(mesh[counter3][0])
                    if NID == mesh[counter3][1]:
                        flag = 0
                        num = counter3
                        DiseaseName = mesh[counter3][0]
                        break
                    counter3 = counter3 + 1

                flag2 = 0
                counter5 = 0
                while counter5 < len(DAGs[counter]):
                    if DAGs[counter][counter5][0] == DiseaseName:
                        flag2 = 1
                        break
                    counter5 = counter5 + 1

                if flag == 0:
                    if flag2 == 0:
                        counter6 = 0
                        IDGroup = []
                        while counter6 < len(mesh):
                            if DiseaseName == mesh[counter6][0]:
                                IDGroup.append(mesh[counter6][1])
                            counter6 = counter6 + 1
                        DiseasePoint = []
                        layer = layer + 1
                        DiseasePoint.append(DiseaseName)
                        DiseasePoint.append(IDGroup)
                        DiseasePoint.append(layer)
                        DAGs[counter].append(DiseasePoint)

            counter2 = counter2 + 1
        counter1 = counter1 + 1
    counter = counter + 1
print('DAGs', len(DAGs))
StorFile(DAGs, 'DAGs2.csv')


DiseaseValue = []
counter = 0
while counter < len(AllDisease):
    if DAGs[counter][0][1] == 0:
        DiseaseValuePair = []
        DiseaseValuePair.append(AllDisease[counter])
        DiseaseValuePair.append(0)
        DiseaseValue.append(DiseaseValuePair)
        counter = counter + 1
        continue
    counter1 = 0
    DV = 0
    while counter1 < len(DAGs[counter]):
        DV = DV + math.pow(0.5, DAGs[counter][counter1][2])
        counter1 = counter1 + 1
    DiseaseValuePair = []
    DiseaseValuePair.append(AllDisease[counter])
    DiseaseValuePair.append(DV)
    DiseaseValue.append(DiseaseValuePair)
    counter = counter + 1
print('len(DiseaseValue)', len(DiseaseValue))
StorFile(DiseaseValue, 'DiseaseValue2.csv')



SameValue1 = []
counter = 0
while counter < len(AllDisease):
    RowValue = []
    if DiseaseValue[counter][1] == 0:
        counter1 = 0
        while counter1 < len(AllDisease):
            RowValue.append(0)
            counter1 = counter1 + 1
        SameValue1.append(RowValue)
        counter = counter + 1
        continue
    counter1 = 0
    while counter1 < len(AllDisease):
        if DiseaseValue[counter1][1] == 0:
            RowValue.append(0)
            counter1 = counter1 + 1
            continue
        DiseaseAndDiseaseSimilarityValue = 0
        counter2 = 0
        while counter2 < len(DAGs[counter]):
            counter3 = 0
            while counter3 < len(DAGs[counter1]):
                if DAGs[counter][counter2][0] == DAGs[counter1][counter3][0]:
                    DiseaseAndDiseaseSimilarityValue = DiseaseAndDiseaseSimilarityValue + math.pow(0.5, DAGs[counter][counter2][2]) + math.pow(0.5, DAGs[counter1][counter3][2]) #自己和自己的全部节点相同，对角线即DiseaseValue的两倍
                counter3 = counter3 + 1
            counter2 = counter2 + 1
        RowValue.append(DiseaseAndDiseaseSimilarityValue)
        counter1 = counter1 + 1
    SameValue1.append(RowValue)
    print(counter)
    counter = counter + 1
print('SameValue1')
StorFile(SameValue1, 'Samevalue12.csv')



DiseaseSimilarityModel1 = []
counter = 0
while counter < len(AllDisease):
    RowValue = []
    if DiseaseValue[counter][1] == 0:
        counter1 = 0
        while counter1 < len(AllDisease):
            RowValue.append(0)
            counter1 = counter1 + 1
        DiseaseSimilarityModel1.append(RowValue)
        counter = counter + 1
        continue
    counter1 = 0
    while counter1 < len(AllDisease):
        if DiseaseValue[counter1][1] == 0:
            RowValue.append(0)
            counter1 = counter1 + 1
            continue
        value = SameValue1[counter][counter1] / (DiseaseValue[counter][1] + DiseaseValue[counter1][1])
        RowValue.append(value)
        counter1 = counter1 + 1
    DiseaseSimilarityModel1.append(RowValue)
    print(counter)
    counter = counter + 1
print('DiseaseSimilarityModel1，', len(DiseaseSimilarityModel1))
print('DiseaseSimilarityModel1[0]', len(DiseaseSimilarityModel1[0]))


counter = 0
while counter < len(DiseaseSimilarityModel1):
    Row = []
    Row.append(AllDisease[counter])
    Row.extend(DiseaseSimilarityModel1[counter])
    DiseaseSimilarityModel1[counter] = Row
    counter = counter + 1


StorFile(DiseaseSimilarityModel1, 'Side_effect_SimilarityModel.csv')

