import numpy as np


def estimateF(x1, x2):
    """
    :param x1: Points from image 1, with shape (coordinates, point_id)
    :param x2: Points from image 2, with shape (coordinates, point_id)
    :return F: Estimated fundamental matrix
    """

    # Use x1 and x2 to construct the equation for homogeneous linear system
    ##-your-code-starts-here-##
    
    a, b= x1[0,:],x1[1,:]
    d,e = x2[0,:],x2[1,:]
    A= np.array([a*d,b*d,d,e*a,e*b,e,a,b])
    A = np.vstack((A,np.ones((1,11))))
    ##-your-code-ends-here-##

    # Use SVD to find the solution for this homogeneous linear system by
    # extracting the row from V corresponding to the smallest singular value.
    ##-your-code-starts-here-##
    u,s,v = np.linalg.svd(A.T)
    ix= np.argmin(s)
    v_min= v[ix]
    ##-your-code-ends-here-##
    F = np.reshape(v_min, (3, 3))  # reshape to acquire Fundamental matrix F
    #F = np.ones((3, 3))  # remove me and uncomment the above

    # Enforce constraint that fundamental matrix has rank 2 by performing
    # SVD and then reconstructing with only the two largest singular values
    # Reconstruction is done with u @ s @ vh where s is the singular values
    # in a diagonal form.
    ##-your-code-starts-here-##
    u,s,v = np.linalg.svd(F)
    s=np.diag(s)
    s[2,2]=0
    recF= u@s@v #should I take u, v also as 2 by 2 or like this??
    ##-your-code-ends-here-##
    
    return recF
