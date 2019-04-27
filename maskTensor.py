
import numpy as np
import random
import os


#get M
def getmaskTensor():
    maskPath =os.path.join( basePath,'input','maskTensor.csv')
    nonZeroIndex = sourceData.nonzero()
    nonzeroNumberofData=sourceData[sourceData>0].size
    M=np.zeros(sourceData.shape,int)
    randomList = random.sample(range(0, nonzeroNumberofData), int(nonzeroNumberofData*0.2))
    for missingIndex in randomList:
        M[nonZeroIndex[0][missingIndex],nonZeroIndex[1][missingIndex],nonZeroIndex[2][missingIndex]]=1
    with open(maskPath, 'w') as fw:
        for data_slice in M:
            fw.write('# New slice\n')
            np.savetxt(fw, data_slice, fmt='%d')
    # print('the number of test sample is:%d'%int(nonzeroNumberofData*0.2))
    return M

def changetrainSet(M):
    # change the size of training set 20%,40%,60%,80%
    trainingSet=sourceData-sourceData*M
    nonZeroIndex = trainingSet.nonzero()
    nonzeroNumberofData = trainingSet[trainingSet>0].size
    for size in [0.2,0.4,0.6,0.8]:
        randomList = random.sample(range(0, nonzeroNumberofData), int(nonzeroNumberofData*size))
        sourceData_size=np.copy(sourceData)
        for missingIndex in randomList:
            sourceData_size[nonZeroIndex[0][missingIndex],nonZeroIndex[1][missingIndex],nonZeroIndex[2][missingIndex]]=0

        sourcesizePath = os.path.join(basePath,'input','rushhour_' + '%.1f' % (1-size)+ '.csv')
        with open(sourcesizePath, 'w') as fw:
            for data_slice in sourceData_size:
                fw.write('# New slice\n')
                np.savetxt(fw, data_slice, fmt='%.6f')

        # missingValue=sourceData_size*M
        # print('the number of sample is:%d' % (sourceData_size[sourceData_size>0].size))
        # print('the number of training set is:%d' % ((sourceData_size[sourceData_size > 0].size)-(missingValue[missingValue>0].size)))

def changetimeSlot():
    sourceData_10Path=os.path.join(basePath,'input','rushHour_10.csv')
    sourceData_15Path = os.path.join(basePath, 'input', 'rushHour_15.csv')
    sourceData_10=np.loadtxt(sourceData_10Path,dtype=float).reshape((n_day,n_road,int(n_interval/2)))
    sourceData_15 = np.loadtxt(sourceData_15Path, dtype=float).reshape((n_day, n_road, int(n_interval / 3)))

    #generate the test set
    nonZeroIndex = sourceData_10.nonzero()
    nonzeroNumberofData = sourceData_10[sourceData_10 > 0].size
    M = np.zeros(sourceData_10.shape, int)
    randomList = random.sample(range(0, nonzeroNumberofData), int(nonzeroNumberofData * 0.2))
    for missingIndex in randomList:
        M[nonZeroIndex[0][missingIndex], nonZeroIndex[1][missingIndex], nonZeroIndex[2][missingIndex]] = 1
    with open(os.path.join(basePath,'input','mask_10.csv'), 'w') as fw:
        for data_slice in M:
            fw.write('# New slice\n')
            np.savetxt(fw, data_slice, fmt='%d')
    # generate the test set
    nonZeroIndex = sourceData_15.nonzero()
    nonzeroNumberofData = sourceData_15[sourceData_15 > 0].size
    M = np.zeros(sourceData_15.shape, int)
    randomList = random.sample(range(0, nonzeroNumberofData), int(nonzeroNumberofData * 0.2))
    for missingIndex in randomList:
        M[nonZeroIndex[0][missingIndex], nonZeroIndex[1][missingIndex], nonZeroIndex[2][missingIndex]] = 1
    with open(os.path.join(basePath,'input','mask_15.csv'), 'w') as fw:
        for data_slice in M:
            fw.write('# New slice\n')
            np.savetxt(fw, data_slice, fmt='%d')

basePath= os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
sourceDataPath=os.path.join(basePath,'input','rushHour.csv')
Mpath=os.path.join(basePath,'input','maskTensor.csv')
n_road=317
n_interval=180
n_day=25
sourceData=np.loadtxt(sourceDataPath,dtype=float).reshape((n_day,n_road,n_interval))

#generate the validation set
maskTensor=getmaskTensor()
#generate the different sizes of training sets
changetrainSet(maskTensor)
#generate the validation sets for the datasets of other slot sizes
changetimeSlot()



