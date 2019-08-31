import numpy as np
from change.processData import ProcessData
import os
from change.variate import Variate

class Init:
    def unfold(self,A,n):
        if type(A) != type(np.zeros((1))):
            print("Error: Function designed to work with numpy ndarrays")
            raise ValueError
        if not (1 <= n <= A.ndim):
            print("Error: axis %d not in range 1:%d" % (n, A.ndim))
            raise np.linalg.LinAlgError
        s = A.shape
        m = 1
        for i in range(len(s)):
            m *= s[i]
        m = int(m / s[n - 1])
        Au = np.zeros((s[n - 1], m))
        index = [0] * len(s)
        for i in range(s[n - 1]):
            index[n - 1] = i
            for j in range(m):
                Au[i, j] = A[tuple(index)]
                index[n - 2] += 1
                for k in range(n - 2, n - 1 - len(s), -1):
                    if index[k] == s[k]:
                        index[k - 1] += 1
                        index[k] = 0

        return Au

    def fold(self,Au, n, s):
        m = 1
        for i in range(len(s)):
            m *= s[i]
        m = int(m / s[n - 1])
        # check for shape compatibility
        if Au.shape != (s[n - 1], m):
            print("Wrong shape: need", (s[n - 1], m), "but have instead", Au.shape)
            raise np.linalg.LinAlgError

        A = np.zeros(s)
        index = [0] * len(s)
        for i in range(s[n - 1]):
            index[n - 1] = i
            for j in range(m):
                A[tuple(index)] = Au[i, j]
                index[n - 2] += 1
                for k in range(n - 2, n - 1 - len(s), -1):
                    if index[k] == s[k]:
                        index[k - 1] += 1
                        index[k] = 0

        return A

    def HOSVD(self,tensor):
        Transforms = []
        # --- Compute the SVD of each possible unfolding
        for i in range(len(tensor.shape)):
            U, D, V = np.linalg.svd(self.unfold(tensor, i + 1))
            Transforms.append(np.asmatrix(U))
        # --- Compute the unfolded core tensor
        axis = 1  # An arbitrary choice, really
        Aun = self.unfold(tensor, axis)
        # --- Computes right hand side transformation matrix
        B = np.ones((1,))
        for i in range(axis - tensor.ndim, axis - 1):
            B = np.kron(B, Transforms[i])
        # --- Compute the unfolded core tensor along the chosen axis
        Sun = Transforms[axis - 1].transpose().conj() * Aun * B
        S = self.fold(Sun, axis, tensor.shape)
        return Transforms, S

def modeProduct(tensor,matrix,moden):
    if moden==0:
        mode_product=np.zeros((matrix.shape[0],tensor.shape[1],tensor.shape[2]),float)
        for k in range(matrix.shape[0]):
            for i in range(tensor.shape[1]):
                for j in range(tensor.shape[2]):
                    mode_product[k,i,j]=matrix[k].dot(tensor[:,i,j])
        return mode_product
    elif moden==1:
        mode_product=np.zeros((tensor.shape[0],matrix.shape[0],tensor.shape[2]),float)
        for k in range(matrix.shape[0]):
            for i in range(tensor.shape[0]):
                for j in range(tensor.shape[2]):
                    mode_product[i,k,j]=matrix[k].dot(tensor[i,:,j])
        return mode_product
    else:
        mode_product=np.zeros((tensor.shape[0],tensor.shape[1],matrix.shape[0]),float)
        for k in range(matrix.shape[0]):
            for i in range(tensor.shape[0]):
                for j in range(tensor.shape[1]):
                    mode_product[i,j,k]=matrix[k].dot(tensor[i,j,:])
        return mode_product

def tdi(A,W,n_emb,lam):
    B = W * A
    lamba = lam
    init=Init()
    matrixList,S_temp=init.HOSVD(B)
    X0= matrixList[0][:,0:n_emb[0]]
    Y0= matrixList[1][:,0:n_emb[1]]
    Z0= matrixList[2][:,0:n_emb[2]]
    S0=S_temp[0:n_emb[0],0:n_emb[1],0:n_emb[2]]
    X = X0.copy()
    Y = Y0.copy()
    Z = Z0.copy()
    S = S0.copy()
    tolerance = 0.1
    iteration = 100
    f_his = np.inf
    for i in range(iteration):
        mp1 = modeProduct(S, X, 0)
        mp2 = modeProduct(mp1, Y, 1)
        mp3 = modeProduct(mp2, Z, 2)
        C = W * mp3
        # update
        temp = modeProduct((C - B), X0.T, 0)
        temp = modeProduct(temp, Y0.T, 1)
        deltaS = modeProduct(temp, Z0.T, 2)
        S = S - lamba * deltaS
        temp = np.kron(Z, Y)
        deltaX = np.dot(np.dot((init.unfold(C, 1) - init.unfold(B, 1)), temp), init.unfold(S, 1).T)
        X = X - lamba * deltaX
        temp = np.kron(Z, X)
        deltaY = np.dot(np.dot((init.unfold(C, 2) - init.unfold(B, 2)), temp), init.unfold(S, 2).T)
        Y = Y - lamba * deltaY
        temp = np.kron(Y, X)
        deltaZ = np.dot(np.dot((init.unfold(C, 3) - init.unfold(B, 3)), temp), init.unfold(S, 3).T)
        Z = Z - lamba * deltaZ
        r = np.sum(np.square(B))
        mp1 = modeProduct(S, X, 0)
        mp2 = modeProduct(mp1, Y, 1)
        mp3 = modeProduct(mp2, Z, 2)
        C = W * mp3
        rcha = np.sqrt(np.sum(np.square(W * (B - C))))
        f = 0.5 * r - np.sum(B * C) + 0.5*np.sum(np.square(C))
        if (rcha / np.sqrt(r)) < tolerance or abs(f - f_his)<1e-3 :
            break
        f_his = f
        if (i != 0 & i < 26):
            if (i < 5):
                lamba = lamba / 2
            if (i % 5 == 0):
                lamba = lamba / 2

    Aestimation = modeProduct(modeProduct(modeProduct(S, X, 0), Y, 1), Z, 2)
    tempAe = Aestimation[M > 0] * 5
    tempA = A[M > 0] * 5
    RMSE = np.sqrt(np.mean(np.square(tempAe - tempA)))
    MAE = np.mean(np.abs(tempAe - tempA))
    return RMSE,MAE

def fold(tensor):
    t_r_list=[]
    for i in range(tensor.shape[1]):
        tensor_road=np.zeros((25,15,int(n_interval/15)))
        matrix_road=tensor[:,i,:]
        for j in range(matrix_road.shape[0]):
            tensor_road[j]=matrix_road[j].reshape(15,int(n_interval/15))
        t_r_list.append(tensor_road)
    return t_r_list

if __name__ == '__main__':
    #init para
    R1=4
    R2=5
    R3=6
    n_emb=[R1,R2,R3]
    maxiter=100
    lam=0.001
    proPath = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
    dataPath = os.path.join(proPath, 'data')
    n_day = 25
    n_road = 317
    n_interval = 180
    # self.n_interval = 90
    # self.n_interval = 60
    A = np.loadtxt(os.path.join(dataPath, 'rushHour.csv'), dtype=float).reshape(n_day, n_road, n_interval)
    M = np.loadtxt(os.path.join(dataPath, 'maskTensor.csv'), dtype=int).reshape(n_day, n_road, n_interval)
    # A = np.loadtxt(os.path.join(dataPath, 'rushHour_10.csv'), dtype=float).reshape(n_day, n_road, n_interval)
    # M = np.loadtxt(os.path.join(dataPath, 'mask_10.csv'), dtype=int).reshape(n_day, n_road, n_interval)
    # A = np.loadtxt(os.path.join(dataPath, 'rushHour_0.8.csv'), dtype=float).reshape(n_day, n_road, n_interval)
    rushhour, mask = fold(A),fold(M)
    resultList=[0,0]
    #if the dimension is too large HOSVD will fail due to insufficient memory
    for road_id in range(len(rushhour)):
        roadTensor=rushhour[road_id]
        M=mask[road_id]
        W = np.ones(M.shape, dtype=int)
        W = W - M
        rmse,mae=tdi(roadTensor,W,n_emb,lam)
        resultList[0]=resultList[0]+rmse
        resultList[1]+=mae
        print('roadid: '+str(road_id))

    print('RMSE: ' + str(round(resultList[0]/len(rushhour),6)))
    print('MAE: ' + str(round(resultList[1] / len(rushhour),6)))