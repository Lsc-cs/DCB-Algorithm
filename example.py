import numpy as np 

import DCB_with_parameter

result = open('./result.txt', 'a')

for e in [3]:
    for m_n in [2000, 5000]:
        for p in [50, 100]:
            for rate in [0.2, 0.8]:
                for t_rate in [1.0]:
                    ATE_list = []
                    ATT_error_list = []
                    for experiment_iter in range(100):
                        print('\n n = %d, p = %d, e = %d, rate = %f, t_rate %f, experiment_iteration = %d......\n'%(m_n, p, e, rate, t_rate, experiment_iter))

                        X = np.random.normal(0, 1, [m_n, p])

                        # set T with linear function (misspecified function)
                        p_t = int(rate * p)
                        T = X[:, :p_t] @ np.ones((p_t, 1)) * t_rate + np.random.normal(0, 1, [m_n, 1])
                        T[T < 0] = 0
                        T[T > 0] = 1

                        # set Y with linear function
                        CE = 1
                        epsilon = np.random.normal(0, e, [m_n, 1])
                        Y = CE * T + epsilon
                        Y1 = CE + epsilon
                        Y0 = epsilon
                        for iter in range(p):
                            weight = 0
                            if np.mod(iter, 2) == 0:
                                weight = (1 + iter) / 2
                            # print(T.shape)
                            # print(X[:, iter].shape)
                            Y += weight * X[:, iter].reshape(m_n, 1) + X[:, iter].reshape(m_n, 1) * T
                            Y1 += weight * X[:, iter].reshape(m_n, 1) + X[:, iter].reshape(m_n, 1)
                            Y0 +=  weight * X[:, iter].reshape(m_n, 1)

                        # ATT ground truth
                        ATT_get = np.mean(Y1[T == 1] - Y0[T == 1])

                        X_t = X[T.flatten() == 1]
                        X_c = X[T.flatten() == 0]
                        Y_t = Y[T.flatten() == 1]
                        Y_c = Y[T.flatten() == 0]

                        m = X_t.shape[0]
                        n = X_c.shape[0]

                        # direct estimator
                        ATE_naive = np.mean(Y_t) - np.mean(Y_c)

                        # DCB estimator
                        W_dcb, ATE_dcb, ATE_dcb_regression = DCB_with_parameter.DCB_parameter(X_t, X_c, Y_t, Y_c)

                        ATT_error = [ATE_naive, ATE_dcb, ATE_dcb_regression] - ATT_get

                        ATT_error_list.append(ATT_error)

                    
                    ATT_error_list = np.array(ATT_error_list).astype(np.float)

                    error_mean = np.mean(ATT_error_list, axis = 0)
                    error_std = np.std(ATT_error_list, axis = 0)
                    error_mae = np.mean(abs(ATT_error_list), axis= 0)
                    error_rmse = np.sqrt(np.mean(ATT_error_list * ATT_error_list, axis=0))

                    result.write('\n m_n: %d --- p: %d --- e: %d --- rate: %f --- t_rate: %f \n'%(m_n, p, e, rate, t_rate))
                    result.write('ATE_naive, ATE_dcb, ATE_dcb_regression \n')
                    result.write('bias: %s\n'%('   '.join(str(e) for e in error_mean)))
                    result.write('std : %s\n'%('   '.join(str(e) for e in error_std)))
                    result.write('MAE : %s\n'%('   '.join(str(e) for e in error_mae)))
                    result.write('RMSE: %s\n'%('   '.join(str(e) for e in error_rmse)))

result.close()