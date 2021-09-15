import numpy as np 


import proc


def DCBAlg(X_t, X_c, Y_t, Y_c, lambda0, lambda1, lambda2, lambda3, lambda4, maxiter, absol):
    '''
    estimate the ATE from observational data via Differentiated Balancing Algorithm
    the objective funtion as flllows:
    J = lambda0 * (beta.T @ (mean_X_t - X_c.T @ W * W)) * (beta.T @ (mean_X_t - X_c.T @ W * W)) + lambda1 * sum((Y_c - X_c @ beta)^2) + lambda2 * (w * w).T @ (W * W) + lambda3 * sum(beta^2) + lambda4* abs(beta)
    Args:
        param X_t: observation variables of treated units
        param X_c: observation variables of control units
        param Y_t: observed outcom of treated units
        param Y_c: observed outcome of control units
        param lambda : hyper-parameters
        param maxiter: the max number of iteration
        param absol: the condition of break
    return:
        ATT, W, beta
    '''

    def f_x(W, beta, mean_X_t, X_c, Y_c, lambda0, lambda1, lambda2, lambda3):
        return lambda0 * (beta.T @ (mean_X_t - X_c.T @ (W * W))) * (beta.T @ (mean_X_t - X_c.T @ (W * W))) \
            + lambda1 * (1 + W * W) .T @ ((Y_c - X_c @ beta) * (Y_c - X_c @ beta)) \
            + lambda2 * ((W * W).T @ (W * W)) \
            + lambda3 * np.sum(np.square(beta))

    m = X_t.shape[0]
    n = X_c.shape[0]
    if X_c.shape[1] == X_t.shape[1]:
        p = X_t.shape[1]
    else: 
        print('error: Dimensions dismatch')
    mean_X_t = np.mean(X_t, axis = 0).reshape(p, 1)

    # parameters initialization
    W = np.ones((n, 1)) / n
    W_prev = W
    beta = np.ones((p, 1)) / p
    beta_prev = beta

    parameter_iter = 0.5
    J_loss = np.ones(maxiter) * -1

    lambda_W = 1     
    lambda_beta = 1 #近端梯度下降步长

    # proximal gradient algorithm
    for iter in range(maxiter):
        # beta
        y = beta
        beta += (iter / (3 + iter)) * (beta - beta_prev)
        f_base = f_x(W, beta, mean_X_t, X_c, Y_c, lambda0, lambda1, lambda2, lambda3)
        while True:
            grad_beta =  2* lambda0 * (beta.T @ (mean_X_t - X_c.T @ (W * W))) * (mean_X_t - X_c.T @ (W * W)) \
                - 2 * lambda1 * X_c.T @ ((1 + W * W) * (Y_c - X_c @ beta)) \
                + 2 * lambda3 * beta
            z = proc.proc_l1(beta - lambda_beta * grad_beta, lambda_beta * lambda4)
            if f_x(W, z, mean_X_t, X_c, Y_c, lambda0, lambda1, lambda2, lambda3) <= f_base + grad_beta.T @ (z - beta) + (1 / (2 * lambda_beta)) * np.sum(np.square(z - beta)):
                break
            lambda_beta = parameter_iter * lambda_beta
        beta_prev = y
        beta = z


        # W
        y = W
        W += (iter / (iter + 3)) * (W - W_prev)
        f_base = f_x(W, beta, mean_X_t, X_c, Y_c, lambda0, lambda1, lambda2, lambda3)
        while True:
            grad_W = -4 * lambda0 * (beta.T @ (mean_X_t - X_c.T @ (W * W))) * X_c @ beta * W\
                + 2 * lambda1 * W * (Y_c - X_c @ beta)\
                + 4 * lambda2 * (W * W) * W
            z = proc.proc_l1(W - lambda_W * grad_W, 0)
            if f_x(z, beta, mean_X_t, X_c, Y_c, lambda0, lambda1, lambda2, lambda3) <= f_base + grad_W.T @ (z - W) + (1 / (2 * lambda_W)) * np.sum(np.square(z - W)):
                break
            lambda_W = parameter_iter * lambda_W
        W_prev = y
        W = z
        W = W / np.sqrt(W.T @ W)


        ATT = np.mean(Y_t) - (W * W).T @ Y_c

        J_loss[iter] = f_x(W, beta, mean_X_t, X_c, Y_c, lambda0, lambda1, lambda2, lambda3) + np.sum(np.abs(beta))

        if iter > 0 and abs(J_loss[iter] - J_loss[iter - 1]) < absol:
            print('Get the optimal results as iteration %d, J_error: %f\n'%(iter, J_loss[iter])) 
            break
        elif iter == maxiter - 1:
            print('J_error: %f\n'%(J_loss[iter])) 
    return ATT, W, beta






