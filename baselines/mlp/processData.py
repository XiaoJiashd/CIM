import os
import numpy as np

def processData(datawithmissing,windowSize):
    train_roadList=[]
    for i in range(datawithmissing.shape[0]):
        tempdata=datawithmissing[i]
        if i==0:
            trainLabel=tempdata[tempdata>0]
        else:
            trainLabel=np.hstack((trainLabel,tempdata[tempdata>0]))
        trainPosition=tempdata.nonzero()[0]
        for j,pos in enumerate(trainPosition):
            tempTrain=np.append(tempdata[pos-windowSize:pos],tempdata[pos+1:pos+windowSize+1])
            if j==0:
                trainData_temp=tempTrain
            else:
                trainData_temp=np.vstack((trainData_temp,tempTrain))
        train_roadList.append(trainData_temp)
    trainData=train_roadList[0]
    for train in train_roadList[1:]:
        trainData=np.vstack((trainData,train))
    return trainData,trainLabel

def processData_test(test_data,windowSize,test_dataLabel):
    test_roadList=[]
    for i in range(test_data.shape[0]):
        tempTestData=test_data[i]
        tempTestLabel=test_dataLabel[i]
        position=np.nonzero(tempTestLabel)[0]
        for j,pos in enumerate(position):
            tempTest=np.append(tempTestData[pos-windowSize:pos],tempTestData[pos+1:pos+windowSize+1])
            if j==0:
                testData_temp=tempTest
            else:
                testData_temp=np.vstack((testData_temp,tempTest))
        test_roadList.append(testData_temp)
    testData=test_roadList[0]
    for test in test_roadList[1:]:
        testData=np.vstack((testData,test))
    return testData

n_day = 25
n_road = 317
n_interval = 180
# n_interval=90
# n_interval=60

proPath = os.path.abspath(os.path.join(os.path.dirname('__file__'), '../..'))
dataPath = os.path.join(proPath, 'data')
rushHourPath = os.path.join(dataPath, 'rushHour.csv')
MPath = os.path.join(dataPath, 'maskTensor.csv')
# rushHourPath = os.path.join(dataPath, 'rushHour_10.csv')
# MPath = os.path.join(dataPath, 'mask_10.csv')
# rushHourPath = os.path.join(dataPath, 'rushHour_0.8.csv')

data = np.loadtxt(rushHourPath, dtype=float).reshape((n_day,n_road,n_interval))
M = np.loadtxt(MPath, dtype=int).reshape((n_day,n_road,n_interval))
datawithmissing = data * (1-M)
dataMissing = data * M
datawithMissing_2=datawithmissing[0]
dataMissing_2=dataMissing[0]
for i in range(1,n_day):
    datawithMissing_2=np.hstack((datawithMissing_2,datawithmissing[i]))
    dataMissing_2=np.hstack((dataMissing_2,dataMissing[i]))
window_size=12
#padding
padding=np.zeros((n_road,window_size))
datawithMissing_2=np.hstack((padding,datawithMissing_2))
datawithMissing_2=np.hstack((datawithMissing_2,padding))
dataMissing_2=np.hstack((padding,dataMissing_2))
dataMissing_2=np.hstack((dataMissing_2,padding))

trainData,trainLabel=processData(datawithMissing_2,window_size)
testData=processData_test(datawithMissing_2,window_size,dataMissing_2)
testLabel=dataMissing_2[dataMissing_2>0]
testLabel=testLabel.reshape(testLabel.size,1)
with open(os.path.join(dataPath,'trainSet_mlp.csv'), 'w')as fw_traindata:
    np.savetxt(fw_traindata,trainData,fmt='%.6f')
with open(os.path.join(dataPath, 'trainLabel_mlp.csv'), 'w')as fw_trainlabel:
    np.savetxt(fw_trainlabel,trainLabel,fmt='%.6f')
with open(os.path.join(dataPath, 'testSet_mlp.csv'), 'w')as fw_testdata:
    np.savetxt(fw_testdata,testData,fmt='%.6f')
with open(os.path.join(dataPath, 'testLabel_mlp.csv'), 'w')as fw_testlabel:
    np.savetxt(fw_testlabel,testLabel, fmt='%.6f')


