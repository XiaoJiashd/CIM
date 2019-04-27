import numpy as np
from scipy import sparse
import os

from cim import Com_Embedding_RC

basePath= os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'data'))
inputfile=os.path.join(basePath,'input')
n_road=317
n_day=25

sourceDataPath=os.path.join(inputfile,'rushHour.csv')
maskCodePath=os.path.join(inputfile,'maskTensor.csv')
n_interval=180
# sourceDataPath=os.path.join(inputfile,'rushHour_10.csv')
# maskCodePath=os.path.join(inputfile,'mask_10.csv')
# n_interval=90
# sourceDataPath=os.path.join(inputfile,'rushHour_15.csv')
# maskCodePath=os.path.join(inputfile,'mask_15.csv')
# n_interval=60

sourceData=np.loadtxt(sourceDataPath,dtype=float)
M=np.loadtxt(maskCodePath,dtype=int)
missing=sourceData*M
D=(sourceData-missing).reshape((n_day,n_road,n_interval))
missingValue=missing.reshape((n_day,n_road,n_interval))

updatePList=[]
updateQList=[]
updateSList=[]
updateTList=[]

for date in range(n_day):
    DMatrix=D[date]

    updatePList.append(sparse.csr_matrix(DMatrix))
    updateQList.append(sparse.csr_matrix(DMatrix.T))

for roadid in range(n_road):
    DMatrix2=D[:,roadid,:]

    updateSList.append(sparse.csr_matrix(DMatrix2.T))
    updateTList.append(sparse.csr_matrix(DMatrix2))

n_embeddings1=15
n_embeddings2=5
beta=0.6

max_iter=150
n_jobs=4
c0=0.01
c1=1
lam=1e-1
lamda_road=0
lamda_day=0
windowSize=4

decomposition=Com_Embedding_RC.Embedding(n_embeddings1=n_embeddings1,n_embeddings2=n_embeddings2,
                                      max_iter=max_iter,batch_size=51,init_std=0.01,n_jobs=n_jobs,
                                      random_state=98765,lam=lam,lamda_road=lamda_road,lamda_day=lamda_day,windowSize=windowSize,beta=beta,c0=c0,c1=c1)

decomposition.fit(updatePList,updateQList,updateSList,updateTList,M,missingValue,D)
