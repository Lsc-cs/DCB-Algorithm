import numpy as np 

def proc_l1(v, Lambda):
    '''
    the function proc_l1 is the proximal operator of the l1 norm to get the soft thresholding
    :param v
    :param Lambda: the coffecient of the l1 norm
    :return : the soft thresholding
    '''
    x = np.maximum(0, v - Lambda) - np.maximum(0, -v - Lambda)
    return x