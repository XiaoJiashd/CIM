import os
import numpy as np

def processData(datawithmissing,window_size,daywin_size):
    trainLabelMatrix=datawithmissing[datawithmissing>0]
    roadtrainLabelList.append(trainLabelMatrix.reshape(trainLabelMatrix.size,1))
    position= datawithmissing.nonzero()
    for i in range(position[0].size):
        posLabel=(position[0][i],position[1][i])
        left_image=datawithmissing[posLabel[0]-daywin_size+1:posLabel[0]+1,posLabel[1]-window_size:posLabel[1]]
        right_image = datawithmissing[posLabel[0] - daywin_size + 1:posLabel[0]+1, posLabel[1] +1:posLabel[1]+window_size+1]
        sample=np.hstack((left_image,right_image))
        roadtrainDataList.append(sample)

def processData_test(datawithmissingMatrix,missingMatrix,window_size,daywin_size):
    testLabelMatrix=missingMatrix[missingMatrix>0]
    roadtestLabelList.append(testLabelMatrix.reshape(testLabelMatrix.size,1))
    position=missingMatrix.nonzero()
    for i in range(position[0].size):
        posLabel=(position[0][i],position[1][i])
        left_image=datawithmissingMatrix[posLabel[0]-daywin_size+1:posLabel[0]+1,posLabel[1]-window_size:posLabel[1]]
        right_image = datawithmissingMatrix[posLabel[0] - daywin_size + 1:posLabel[0]+1, posLabel[1] +1:posLabel[1]+window_size+1]
        sample=np.hstack((left_image,right_image))
        roadtestDataList.append(sample)

n_day = 25
n_road = 317
n_intervel = 180
# n_intervel = 90
# n_intervel = 60

proPath = os.path.abspath(os.path.join(os.path.dirname('__file__'), '../..'))
dataPath = os.path.join(proPath, 'data')
rushHourPath = os.path.join(dataPath, 'rushHour.csv')
MPath = os.path.join(dataPath, 'maskTensor.csv')
# rushHourPath = os.path.join(dataPath, 'rushHour_10.csv')
# MPath = os.path.join(dataPath, 'mask_10.csv')
# rushHourPath = os.path.join(dataPath, 'rushHour_0.8.csv')

data = np.loadtxt(rushHourPath, dtype=float).reshape(n_day,n_road,n_intervel)
M = np.loadtxt(MPath, dtype=int).reshape(n_day,n_road,n_intervel)
datawithmissing = data * (1-M)
dataMissing = data * M
daywin_size=5
window_size=12
#padding
roadtrainDataList=[]
roadtrainLabelList=[]
roadtestDataList=[]
roadtestLabelList=[]
padding_left=np.zeros((n_day,window_size))
padding_head=np.zeros((daywin_size-1,n_intervel+2*window_size))
for road in range(n_road):
    roadMatrix=np.hstack((padding_left,datawithmissing[:,road,:]))
    roadMatrix=np.hstack((roadMatrix,padding_left))
    roadMatrix=np.vstack((padding_head,roadMatrix))
    missingMatrix=np.hstack((padding_left,dataMissing[:,road,:]))
    missingMatrix=np.hstack((missingMatrix,padding_left))
    missingMatrix=np.vstack((padding_head,missingMatrix))
    processData(roadMatrix,window_size,daywin_size)
    processData_test(roadMatrix,missingMatrix,window_size,daywin_size)

trainData=np.array(roadtrainDataList)
trainLabel=roadtrainLabelList[0]
testData=np.array(roadtestDataList)
testLabel=roadtestLabelList[0]
for i in range(1,n_road):
    trainLabel=np.vstack((trainLabel,roadtrainLabelList[i]))
    testLabel = np.vstack((testLabel, roadtestLabelList[i]))
with open(os.path.join(dataPath,'trainSet_cnn.csv'), 'w')as fw_traindata:
    for data_slice in trainData:
        np.savetxt(fw_traindata,data_slice,fmt='%.6f')
with open(os.path.join(dataPath, 'trainLabel_cnn.csv'), 'w')as fw_trainlabel:
    np.savetxt(fw_trainlabel,trainLabel,fmt='%.6f')
with open(os.path.join(dataPath, 'testSet_cnn.csv'), 'w')as fw_testdata:
    for data_slice in testData:
        np.savetxt(fw_testdata,data_slice,fmt='%.6f')
with open(os.path.join(dataPath, 'testLabel_cnn.csv'), 'w')as fw_testlabel:
    np.savetxt(fw_testlabel,testLabel, fmt='%.6f')