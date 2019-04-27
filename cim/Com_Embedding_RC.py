import time
import numpy as np
from numpy import linalg as LA
from joblib import Parallel,delayed
from sklearn.base import  BaseEstimator,TransformerMixin
import os

basePath= os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'data'))
class Embedding(BaseEstimator,TransformerMixin):
    def __init__(self,n_embeddings1=100,n_embeddings2=10,max_iter=50,batch_size=51,
                 init_std=0.01,dtype='float32',n_jobs=8,windowSize=3,random_state=None,**kwargs):
        self.n_embeddings1=n_embeddings1
        self.n_embeddings2=n_embeddings2
        self.max_iter=max_iter
        self.batch_size=batch_size
        self.init_std=init_std
        self.dtype=dtype
        self.n_jobs=n_jobs
        self.windowSize=windowSize
        self.random_state=random_state

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.set_state(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        '''model hyperparameters'''
        self.lam = float(kwargs.get('lam', 1e-5))
        self.lamda_road=float(kwargs.get('lamda_road',1e-2))
        self.lamda_day=float(kwargs.get('lamda_day',1e-2))
        self.beta=float(kwargs.get('beta',1.0))
        self.c0 = float(kwargs.get('c0', 0.01))
        self.c1 = float(kwargs.get('c1', 1.0))
        assert self.c0 <= self.c1, "c0must be smaller than c1"

    def _init_params(self, n_road, n_interval,n_day):
        ''' Initialize all the latent factors and biases '''
        self.roadP=self.init_std*np.random.randn(n_day,n_road,self.n_embeddings1).astype(self.dtype)+1
        self.intervalQ=self.init_std*np.random.randn(n_day,n_interval,self.n_embeddings1).astype(self.dtype)+1
        self.intervalS=self.init_std*np.random.randn(n_road,n_interval,self.n_embeddings2).astype(self.dtype)+1
        self.dayT=self.init_std*np.random.randn(n_road,n_day,self.n_embeddings2).astype(self.dtype)+1

        assert np.all(self.roadP>0)
        assert np.all(self.intervalQ>0)
        assert np.all(self.intervalS>0)
        assert np.all(self.dayT>0)

    def fit(self,XPList,XQList,XSList,XTList,M,missingValue,D):
        n_road,n_interval=XPList[0].shape
        n_day=len(XPList)

        self._init_params(n_road,n_interval,n_day)
        start_t=time.time()
        self._update(XPList,XQList,XSList,XTList,M,missingValue,D)
        # print('TIME #%.2f'%(time.time()-start_t))
        return self

    def _update(self,XPList,XQList,XSList,XTList,M,missingValue,D):
        shape=(len(XPList),XPList[0].shape[0],XPList[0].shape[1])
        nnz=len(missingValue.nonzero()[0])
        temploss = 0

        for i in range(self.max_iter):
            self._update_factors(XPList,XQList,XSList,XTList)

            tempMatrixPQ = np.empty(shape, dtype=float)
            tempMatrixST = np.empty(shape, dtype=float)

            for iterday in range(shape[0]):
                tempMatrixPQ[iterday]=np.dot(self.roadP[iterday],self.intervalQ[iterday].T)
            for iterRoad in range(shape[1]):
                tempMatrixST[:,iterRoad,:]=np.dot(self.dayT[iterRoad],self.intervalS[iterRoad].T)

            loss=self.beta*(np.sum(np.square(D[D > 0] - tempMatrixPQ[D > 0])))+\
                 (1-self.beta)*(np.sum(np.square(D[D>0]-tempMatrixST[D>0])))+\
                 self.lam*(np.sum(np.square(self.roadP))+np.sum(np.square(self.intervalQ))+np.sum(np.square(self.intervalS))+np.sum(np.square(self.dayT)))
            # print('ITERATION #%d' % i, loss)
            if abs(loss-temploss)<0.1:
                break
            temploss=loss

        tempMatrix = (self.beta* tempMatrixPQ + (1 - self.beta) * tempMatrixST)*M.reshape(shape) #reshape(25,317,90)reshape(25,317,60)
        completeMatrixPath=os.path.join(basePath,'output','missValue.csv')
        with open(completeMatrixPath,'w')as cmfw:
            for data_slice in tempMatrix:
                cmfw.write("#new slice\n")
                np.savetxt(cmfw,data_slice,fmt='%.6f')
        tempMatrix=tempMatrix*5
        missingValuetemp=missingValue*5

        MAE = np.sum(np.abs(tempMatrix[missingValuetemp > 0] - missingValuetemp[missingValuetemp > 0])) / nnz
        MAPE = np.sum(np.abs(tempMatrix[missingValuetemp > 0] - missingValuetemp[missingValuetemp > 0]) / missingValuetemp[missingValuetemp > 0]) / nnz
        RMSE = np.sqrt(np.sum(np.square(tempMatrix[missingValuetemp > 0] - missingValuetemp[missingValuetemp > 0])) / nnz)
        with open( os.path.join(basePath,'output','results'), 'a')as refw:
            refw.write('RMSE:%.6f\n' % RMSE )
            refw.write('MAPE:%.6f\n' % MAPE)
            refw.write('MAE:%.6f\n' % MAE)

    def _update_factors(self,XPList,XQList,XSList,XTList):
        self.roadP=update_roadP(self.intervalQ,XPList,self.beta,self.c0,self.c1,self.lam,self.lamda_road,self.windowSize,self.n_jobs,self.batch_size)
        self.intervalQ = update_intervalQ(self.roadP,self.intervalQ, XQList,self.beta, self.c0, self.c1, self.lam,self.lamda_road,self.windowSize, self.n_jobs, self.batch_size)
        self.intervalS = update_intervalS(self.dayT,self.intervalS, XSList, self.beta, self.c0, self.c1, self.lam,self.lamda_day,self.windowSize, self.n_jobs,self.batch_size)
        self.dayT= update_dayT(self.intervalS, XTList, self.beta, self.c0, self.c1, self.lam,self.lamda_day,self.windowSize, self.n_jobs,self.batch_size)

def get_row(Y,i):
    lo,hi=Y.indptr[i],Y.indptr[i+1]
    return Y.data[lo:hi],Y.indices[lo:hi]

def update_roadP(intervalQ,XPList,beta,c0,c1,lam,lamda_road,w,n_jobs,batch_size):
    m,n=XPList[0].shape
    updateIter=len(XPList)
    f=intervalQ.shape[2]
    assert updateIter==intervalQ.shape[0]

    roadP=np.empty((updateIter,m,f),dtype=float)

    for iter in range(updateIter):
        interval=intervalQ[iter]
        mean_W = np.empty(interval.shape,float)
        for rowid in range(mean_W.shape[0]):
            if(rowid<w):
                mean_W[rowid]=np.sum(interval[:rowid+w+1],axis=0)
            elif(rowid>=w and rowid<mean_W.shape[0]-w):
                mean_W[rowid]=np.sum(interval[rowid-w:rowid+w+1],axis=0)
            else:
                mean_W[rowid] = np.sum(interval[rowid - w:], axis=0)
        mean_W=mean_W/2/w
        diffMatrix=interval-mean_W

        BTB=c0*np.dot(interval.T,interval)
        DTD=c0*np.dot(diffMatrix.T,diffMatrix)
        BTBpR=BTB+lamda_road*DTD+lam*np.eye(f,dtype=interval.dtype)#beta*

        start_idx=list(range(0,m,batch_size))
        end_idx=start_idx[1:]+[m]
        res=Parallel(n_jobs)(delayed(_solve_roadP)(lorow,hirow,interval,XPList[iter],BTBpR,diffMatrix,beta,lamda_road,c0,c1,f)for lorow,hirow in zip(start_idx,end_idx))
        roadP[iter]=np.vstack(res)
    return roadP

def _solve_roadP(lorow,hirow,interval,X,BTBpR,diffMatrix,beta,lamda_road,c0,c1,f):
    road_batch = np.empty((hirow - lorow, f), dtype=interval.dtype)
    for ib, u in enumerate(range(lorow, hirow)):
        x_u, idx_u = get_row(X, u)
        B_u = interval[idx_u]
        D_u=diffMatrix[idx_u]
        a =beta*x_u.dot(c1 * B_u)

        B = BTBpR +beta*B_u.T.dot((c1 - c0) * B_u) +lamda_road*D_u.T.dot((c1 - c0) * D_u) #beta*   B_u only contains vectors corresponding to non-zero doc-word occurence
        road_batch[ib] = LA.solve(B, a)
    road_batch=road_batch.clip(0)
    return road_batch

def update_intervalQ(roadP,intervalQ_source,XQList,beta,c0,c1,lam,lamda_road,w,n_jobs,batch_size):
    m,n=XQList[0].shape
    updateIter=len(XQList)
    f=roadP.shape[2]
    assert updateIter == roadP.shape[0]

    intervalQ = np.empty((updateIter, m, f), dtype=float)

    for iter in range(updateIter):
        road = roadP[iter]
        interval_source=intervalQ_source[iter]

        BTB = c0 * np.dot(road.T, road)
        DTD=c0*np.dot(road.T,road)
        BTBpR =BTB +lamda_road*DTD+ lam * np.eye(f, dtype=road.dtype)

        start_idx = list(range(0, m, batch_size))
        end_idx = start_idx[1:] + [m]
        res = Parallel(n_jobs)(delayed(_solve_intervalQ)(lorow, hirow, road,interval_source, XQList[iter], BTBpR, beta, lamda_road,w,c0, c1, f) for lorow, hirow in
            zip(start_idx, end_idx))
        intervalQ[iter] = np.vstack(res)
    return intervalQ

def _solve_intervalQ(lorow,hirow,road,interval_source,XT,BTBpR,beta,lamda_road,w,c0,c1,f):
    interval_sbackup=np.copy(interval_source)
    interval_batch = np.empty((hirow - lorow, f), dtype=road.dtype)
    mean_w=np.empty((1,XT.shape[0]),dtype=float)

    for ib, u in enumerate(range(lorow, hirow)):
        mean_Qj=np.empty((1,interval_batch.shape[1]),float)
        if (u < w):
            mean_Qj = np.sum(interval_sbackup[:u + w + 1], axis=0)
        elif (u >= w and u < interval_sbackup.shape[0] - w):
            mean_Qj = np.sum(interval_sbackup[u - w:u + w + 1], axis=0)
        else:
            mean_Qj = np.sum(interval_sbackup[u - w:], axis=0)
        mean_Qj=mean_Qj/2/w
        mean_w=road.dot(mean_Qj)

        x_u, idx_u = get_row(XT, u)
        B_u = road[idx_u]
        mean_w_u=mean_w[idx_u]
        a = beta * x_u.dot(c1 * B_u)
        Da=mean_w_u.dot(B_u)
        a=a+lamda_road*Da

        B = BTBpR +(beta+lamda_road) * B_u.T.dot((c1 - c0) * B_u)
        interval_batch[ib] = LA.solve(B, a)
        interval_sbackup[u]=interval_batch[ib].clip(0)

    interval_batch = interval_batch.clip(0)
    return interval_batch

def update_intervalS(dayT,interval_source,XSList,beta,c0,c1,lam,lamda_day,w,n_jobs,batch_size):
    m,n=XSList[0].shape
    updateIter=len(XSList)
    f=dayT.shape[2]
    assert updateIter == dayT.shape[0]

    intervalS = np.empty((updateIter, m, f), dtype=float)

    for iter in range(updateIter):
        day = dayT[iter]
        interval=interval_source[iter]
        BTB = c0 *np.dot(day.T, day)
        DTD=c0 *np.dot(day.T,day)
        BTBpR =  BTB + lamda_day*DTD+lam * np.eye(f, dtype=day.dtype)#(1-beta) *

        start_idx = list(range(0, m, batch_size))
        end_idx = start_idx[1:] + [m]
        res = Parallel(n_jobs)(delayed(_solve_intervalS)(lorow, hirow, day,interval, XSList[iter], BTBpR, beta,lamda_day,w, c0, c1, f) for lorow, hirow in
            zip(start_idx, end_idx))
        intervalS[iter] = np.vstack(res)
    return intervalS

def _solve_intervalS(lorow,hirow,day,interval,XT,BTBpR,beta,lamda_day,w,c0,c1,f):
    interval_sbackup=np.copy(interval)
    interval_batch = np.empty((hirow - lorow, f), dtype=day.dtype)

    mean_w = np.empty((1, XT.shape[0]), dtype=float)
    for ib, u in enumerate(range(lorow, hirow)):

        mean_Sj = np.empty((1, interval_batch.shape[1]), float)
        if (u < w):
            mean_Sj = np.sum(interval_sbackup[:u + w + 1], axis=0)
        elif (u >= w and u < interval_sbackup.shape[0] - w):
            mean_Sj = np.sum(interval_sbackup[u - w:u + w + 1], axis=0)
        else:
            mean_Sj = np.sum(interval_sbackup[u - w:], axis=0)
        mean_Sj = mean_Sj / 2 / w
        mean_w = day.dot(mean_Sj)

        x_u, idx_u = get_row(XT, u)
        B_u = day[idx_u]
        mean_w_u=mean_w[idx_u]
        a = (1-beta) *x_u.dot(c1 * B_u)
        Da=mean_w_u.dot(B_u)
        a=a+lamda_day*Da

        B = BTBpR + (1-beta+lamda_day) * B_u.T.dot((c1 - c0)* B_u)
        interval_batch[ib] = LA.solve(B, a)
        interval_sbackup[u]=interval_batch[ib].clip(0)
    interval_batch = interval_batch.clip(0)
    return interval_batch

def update_dayT(intervalS,XTList,beta,c0,c1,lam,lamda_day,w,n_jobs,batch_size):
    m,n=XTList[0].shape
    updateIter=len(XTList)
    f=intervalS.shape[2]
    assert updateIter == intervalS.shape[0]

    dayT = np.empty((updateIter, m, f), dtype=float)

    for iter in range(updateIter):
        interval = intervalS[iter]
        mean_W = np.empty(interval.shape, float)
        for rowid in range(mean_W.shape[0]):
            if (rowid < w):
                mean_W[rowid] = np.sum(interval[:rowid + w + 1], axis=0)
            elif (rowid >= w and rowid < mean_W.shape[0] - w):
                mean_W[rowid] = np.sum(interval[rowid - w:rowid + w + 1], axis=0)
            else:
                mean_W[rowid] = np.sum(interval[rowid - w:], axis=0)
        mean_W = mean_W / 2 / w
        diffMatrix = interval - mean_W

        X=XTList[iter]
        BTB =  c0 *np.dot(interval.T, interval)#c0 *
        DTD=c0*np.dot(diffMatrix.T,diffMatrix)
        BTBpR =  BTB +lamda_day*DTD+ lam * np.eye(f, dtype=interval.dtype)

        day=np.empty((m,f),dtype=interval.dtype)
        for ib, u in enumerate(range(m)):
            x_u, idx_u = get_row(X, u)
            B_u = interval[idx_u]
            D_u=diffMatrix[idx_u]
            a = (1 - beta) * x_u.dot(c1 * B_u)

            B = BTBpR + (1 - beta) *B_u.T.dot((c1 - c0)* B_u) +lamda_day *D_u.T.dot((c1 - c0)* D_u) #(c1 - c0) *# B_u only contains vectors corresponding to non-zero doc-word occurence
            day[ib] = LA.solve(B, a)
        dayT[iter]=day.clip(0)
    return dayT