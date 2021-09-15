import numpy as np 

import DCB

def DCB_parameter(X_t, X_c, Y_t, Y_c):
    '''
    differentied confounder balancing
    Args:
        X_t: observation variables of treated units
        X_c: obsercation variables of control units
        Y_t: outcome of treated units
        Y_c: ourcome of control units
    return:
        W: the sample weights
        ATE: ATT
        ATE_residual:
    '''

    lambda0 = 10
    lambda1 = 10
    lambda2 = 100
    lambda3 = 1
    lambda4 = 0.1
    maxiter = 1000
    absol = 1e-6

    ATE, W, beta = DCB.DCBAlg(X_t, X_c, Y_t, Y_c, lambda0, lambda1, lambda2, lambda3, lambda4, maxiter, absol)

    W = W * W

    ATE_residual = np.mean(Y_t) - (np.mean(X_t, axis= 0) @ beta + W.T @ (Y_c - X_c @ beta))

    return W, ATE, ATE_residual