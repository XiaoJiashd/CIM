import numpy as np
from numpy.random import multivariate_normal as mvnrnd
from scipy.stats import wishart
from numpy.linalg import inv as inv
import os
import math

def kr_prod(a, b):
    return np.einsum('ir, jr -> ijr', a, b).reshape(a.shape[0] * b.shape[0], -1)

def cp_combine(U, V, X):
    return np.einsum('is, js, ts -> ijt', U, V, X)

def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

def cov_mat(mat):
    dim1, dim2 = mat.shape
    new_mat = np.zeros((dim2, dim2))
    mat_bar = np.mean(mat, axis = 0)
    for i in range(dim1):
        new_mat += np.einsum('i, j -> ij', mat[i, :] - mat_bar, mat[i, :] - mat_bar)
    return new_mat

def BGCP(dense_tensor, sparse_tensor, init, rank, maxiter1, maxiter2):
    """Bayesian Gaussian CP (BGCP) decomposition."""
    dim1, dim2, dim3 = sparse_tensor.shape
    binary_tensor = np.zeros((dim1, dim2, dim3))
    dim = np.array([dim1, dim2, dim3])
    pos = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    position = np.where(sparse_tensor != 0)
    binary_tensor[position] = 1
    U = init["U"]
    V = init["V"]
    X = init["X"]
    beta0 = 1
    nu0 = rank
    mu0 = np.zeros((rank))
    W0 = np.eye(rank)
    tau = 1
    alpha = 1e-5
    beta = 1e-5
    U_plus = np.zeros((dim1, rank))
    V_plus = np.zeros((dim2, rank))
    X_plus = np.zeros((dim3, rank))
    tensor_hat_plus = np.zeros((dim1, dim2, dim3))
    for iters in range(maxiter1):
        for order in range(dim.shape[0]):
            if order == 0:
                mat = U.copy()
            elif order == 1:
                mat = V.copy()
            else:
                mat = X.copy()
            mat_bar = np.mean(mat, axis=0)
            var_mu_hyper = (dim[order] * mat_bar + beta0 * mu0) / (dim[order] + beta0)
            var_W_hyper = inv(inv(W0) + cov_mat(mat) + dim[order] * beta0 / (dim[order] + beta0)
                              * np.outer(mat_bar - mu0, mat_bar - mu0))
            var_Lambda_hyper = wishart(df=dim[order] + nu0, scale=var_W_hyper, seed=None).rvs()
            var_mu_hyper = mvnrnd(var_mu_hyper, inv((dim[order] + beta0) * var_Lambda_hyper))
            if order == 0:
                var1 = kr_prod(X, V).T
            elif order == 1:
                var1 = kr_prod(X, U).T
            else:
                var1 = kr_prod(V, U).T
            var2 = kr_prod(var1, var1)
            var3 = (tau * np.matmul(var2, ten2mat(binary_tensor, order).T).reshape([rank, rank, dim[order]])
                    + np.dstack([var_Lambda_hyper] * dim[order]))
            var4 = (tau * np.matmul(var1, ten2mat(sparse_tensor, order).T)
                    + np.dstack([np.matmul(var_Lambda_hyper, var_mu_hyper)] * dim[order])[0, :, :])
            for i in range(dim[order]):
                var_Lambda = var3[:, :, i]
                inv_var_Lambda = inv((var_Lambda + var_Lambda.T) / 2)
                vec = mvnrnd(np.matmul(inv_var_Lambda, var4[:, i]), inv_var_Lambda)
                if order == 0:
                    U[i, :] = vec.copy()
                elif order == 1:
                    V[i, :] = vec.copy()
                else:
                    X[i, :] = vec.copy()
        if iters + 1 > maxiter1 - maxiter2:
            U_plus += U
            V_plus += V
            X_plus += X

        tensor_hat = cp_combine(U, V, X)
        if iters + 1 > maxiter1 - maxiter2:
            tensor_hat_plus += tensor_hat
        rmse = np.sqrt(np.sum((dense_tensor[pos] - tensor_hat[pos]) ** 2) / dense_tensor[pos].shape[0])

        var_alpha = alpha + 0.5 * sparse_tensor[position].shape[0]
        error = sparse_tensor - tensor_hat
        var_beta = beta + 0.5 * np.sum(error[position] ** 2)
        tau = np.random.gamma(var_alpha, 1 / var_beta)

        if (iters + 1) % 200 == 0 and iters < maxiter1 - maxiter2:
            print('Iter: {}'.format(iters + 1))
            print('RMSE: {:.6}'.format(rmse))
            print()
    U = U_plus / maxiter2
    V = V_plus / maxiter2
    X = X_plus / maxiter2
    tensor_hat = tensor_hat_plus / maxiter2
    final_mae = np.sum(np.abs(dense_tensor[pos]*5 - tensor_hat[pos]*5)) / dense_tensor[pos].shape[0]
    final_rmse = np.sqrt(np.sum((dense_tensor[pos]*5 - tensor_hat[pos]*5) ** 2) / dense_tensor[pos].shape[0])
    print('RMSE: {:.6}'.format(final_rmse))
    print('MAE: {:.6}'.format(final_mae))
    print()

proPath=os.path.abspath(os.path.join(os.path.dirname('__file__'),'..'))
dataPath=os.path.join(proPath,'data')
n_day=25
n_road=317
n_interval=180
# n_interval=90
# n_interval=60
X=np.loadtxt(os.path.join(dataPath,'rushHour.csv'),dtype=float).reshape(n_day,n_road,n_interval)
M=np.loadtxt(os.path.join(dataPath,'maskTensor.csv'),dtype=int).reshape(n_day,n_road,n_interval)
# X=np.loadtxt(os.path.join(dataPath,'rushHour_10.csv'),dtype=float).reshape(n_day,n_road,n_interval)
# M=np.loadtxt(os.path.join(dataPath,'mask_10.csv'),dtype=int).reshape(n_day,n_road,n_interval)
# X=np.loadtxt(os.path.join(dataPath,'rushHour_0.8.csv'),dtype=float).reshape(n_day,n_road,n_interval)

sparse_tensor=X*(1-M)
dense_tensor=X
dim=[n_day,n_road,n_interval]
dim1, dim2, dim3 = sparse_tensor.shape
rank = 20
init = {"U": np.random.random((n_day,rank))/math.sqrt(n_day*rank),
        "V": np.random.random((n_road,rank))/math.sqrt(n_road*rank),
       "X": np.random.random((n_interval,rank))/math.sqrt(n_interval*rank)}
maxiter1 = 200
maxiter2 = 150
BGCP(dense_tensor, sparse_tensor, init, rank, maxiter1, maxiter2)

