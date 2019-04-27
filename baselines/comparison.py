import numpy as np
from tensorly.decomposition import tucker
import tensorly as tl
import os

def evaluate(prediction,realValue):
    pretemp=prediction * 5
    realtemp=realValue *5
    errorMatrix=realtemp-pretemp

    RMSE=np.sqrt(np.sum(np.square(errorMatrix))/ testNumber)
    MAE = np.sum(abs(errorMatrix))/ testNumber

    print('the RMSE of '  + ' is:' + str(RMSE))
    print('the MAE of '+' is:' + str(MAE))

def m_MeanValue(depth,M,datawithMissingValue):
    shape = datawithMissingValue.shape
    completeMatrix=np.zeros(datawithMissingValue.shape,float)
    for k in range(shape[0]):
        for i in range(shape[1]):
            for j in  range(shape[2]):
                if(M[k,i,j]==1):
                    if (k < depth):
                        nonZeroNumber = len(np.nonzero(datawithMissingValue[:k+depth+1, i, j])[0])
                        if (nonZeroNumber != 0):
                            completeMatrix[k, i, j] = np.sum(datawithMissingValue[:k+depth+1, i, j]) / nonZeroNumber
                    elif (k > shape[0]-1-depth):
                        nonZeroNumber = len(np.nonzero(datawithMissingValue[k-depth:, i, j])[0])
                        if (nonZeroNumber != 0):
                            completeMatrix[k, i, j] = np.sum(datawithMissingValue[k-depth:, i, j]) / nonZeroNumber
                    else:
                        nonZeroNumber = len(np.nonzero(datawithMissingValue[k-depth:k+depth+1, i, j])[0])
                        if (nonZeroNumber != 0):
                            completeMatrix[k, i, j] = np.sum(datawithMissingValue[k-depth:k+depth+1, i, j ]) / nonZeroNumber
    return completeMatrix

basePath= os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'data'))
shape=(25,317,180)#(25,317,90),(25,317,60)
sourceDataPath=os.path.join(basePath,'input','rushHour.csv')#'rushHour_10.csv','rushHour_15.csv'
maskCodePath=os.path.join(basePath,'input','maskTensor.csv')#'mask_10.csv','mask_15.csv'

sourceData=np.loadtxt(sourceDataPath,dtype=float).reshape(shape)
M=np.loadtxt(maskCodePath,dtype=int).reshape(shape)
missingValue=sourceData*M
datawithMissingValue=sourceData-missingValue

comMatrix=np.zeros(shape,float)
testNumber=len(M[M==1])

print('tucker:')
core_rank=[[5,5,5],[5,10,5],[5,20,5],[5,30,5],[5,30,10],[10,30,10],[10,50,10]]
for rank in core_rank:
    datawithmissing = tl.tensor(datawithMissingValue, dtype='float64')
    core, tucker_factors = tucker(datawithmissing, ranks=rank,n_iter_max=250, init='svd', tol=10e-5)
    comtempMatrix = tl.tucker_to_tensor(core, tucker_factors)
    comMatrix=comtempMatrix*M
    evaluate(comMatrix,missingValue)

print('M_Mean:')
depth=[3,4,5,6,7,8,9,10,11,12]
for arg in depth:
    comMatrix=m_MeanValue(arg,M,datawithMissingValue)
    evaluate(comMatrix*M,missingValue)