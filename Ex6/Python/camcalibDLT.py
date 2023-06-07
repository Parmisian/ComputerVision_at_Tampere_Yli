import numpy as np


def camcalibDLT(x_world, x_im):
    """
    :param x_world: World coordinatesm with shape (point_id, coordinates)
    :param x_im: Image coordinates with shape (point_id, coordinates)
    :return P: Camera projection matrix with shape (3,4)
    """

    # Create the matrix A 
    ##-your-code-starts-here-##
    A = []
    n = x_world.shape[0]
    for i in range(n):
        
        x = x_world[i].reshape(1,-1)
        XT = x_im[i].reshape(-1,1)
        XTx = np.dot(XT,x)
        Z = np.zeros((1,4))
        
        r0 = (-1*XTx[0]).reshape(1,-1)
        r1 = (-1*XTx[1]).reshape(1,-1)
        r2 = XTx[2].reshape(1,-1)
        
        A.append(np.hstack((Z,r2,r1))[0])
        A.append(np.hstack((r2,Z,r0))[0])

    A = np.array(A)
       
    ##-your-code-ends-here-##
    
    # Perform homogeneous least squares fitting.
    # The best solution is given by the eigenvector of
    # A.T*A with the smallest eigenvalue.
    ##-your-code-starts-here-##
    u,s,w = np.linalg.svd(A)
    s_min = np.argmin(s)
    ev= w[s_min]
    ##-your-code-ends-here-##
    
    # Reshape the eigenvector into a projection matrix P
    P = np.reshape(ev, (3, 4))  # here ev is the eigenvector from above
    #P = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=float)  # remove this and uncomment the line above
    
    return P
