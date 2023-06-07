import numpy as np


def trianglin(P1, P2, x1, x2):
    """
    :param P1: Projection matrix for image 1 with shape (3,4)
    :param P2: Projection matrix for image 2 with shape (3,4)
    :param x1: Image coordinates for a point in image 1
    :param x2: Image coordinates for a point in image 2
    :return X: Triangulated world coordinates
    """
    
    # Form A and get the least squares solution from the eigenvector 
    # corresponding to the smallest eigenvalue
    ##-your-code-starts-here-##
    a,b,c= x1
    d,e,f=x2
    ax1= np.array([[0, -c, b],
                   [c,0,-a],
                   [-b,a,0]])
    ax2= np.array([[0,-f,e],
                   [f,0,-d],
                   [-e,d,0]])
    
    xP1= np.dot(ax1,P1)
    xP2=np.dot(ax2,P2)
    A= np.vstack((xP1,xP2))
    
    u,s,w = np.linalg.svd(A)
    s_min = np.argmin(s)
    X = w[s_min]
    
    ##-your-code-ends-here-##
    
    return X
