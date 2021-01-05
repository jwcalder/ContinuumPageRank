import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import scipy.spatial as spatial
import scipy.sparse as sparse
import mayavi.mlab as mlab
from joblib import Parallel, delayed

#u = 2 - (cos(2pix) + cos(2piy))
#v = 2 - (1+gamma_hpi^2/2) (cos(2pix) + cos(2pi y))
#Phi = 1/pi, sigma = 1/4

#Build graph over torus
def torus_eps_graph(X,eps):

    #Create periodic copies of points
    n = X.shape[0]
    I = np.arange(n)
    Y = X.copy()
    J = I.copy()
    x = X[:,0]; y = X[:,1]
    Y = np.vstack((Y,X[x <= eps,:]+[1,0]))
    J = np.hstack((J,I[x <= eps]))
    Y = np.vstack((Y,X[x >= 1-eps,:]-[1,0]))
    J = np.hstack((J,I[x >= 1-eps]))
    Y = np.vstack((Y,X[y <= eps,:]+[0,1]))
    J = np.hstack((J,I[y <= eps]))
    Y = np.vstack((Y,X[y >= 1-eps,:]-[0,1]))
    J = np.hstack((J,I[y >= 1-eps]))
    Y = np.vstack((Y,X[(x >= 1-eps) & (y >= 1-eps),:]-[1,1]))
    J = np.hstack((J,I[(x >= 1-eps) & (y >= 1-eps)]))
    Y = np.vstack((Y,X[(x <= eps) & (y <= eps),:]+[1,1]))
    J = np.hstack((J,I[(x <= eps) & (y <= eps)]))
    Y = np.vstack((Y,X[(x <= eps) & (y >= 1-eps),:]+[1,-1]))
    J = np.hstack((J,I[(x <= eps) & (y >= 1-eps)]))
    Y = np.vstack((Y,X[(y <= eps) & (x >= 1-eps),:]+[-1,1]))
    J = np.hstack((J,I[(y <= eps) & (x >= 1-eps)]))

    #Run range search 
    Xtree = spatial.cKDTree(Y)
    M = Xtree.query_pairs(eps)
    M = np.array(list(M))

    #Symmetrize rangesearch and add diagonal
    M1 = np.concatenate((M[:,0],M[:,1],np.arange(0,n)))
    M2 = np.concatenate((M[:,1],M[:,0],np.arange(0,n)))

    #Restrict to original dataset
    mask = M1 < n
    M1 = M1[mask]
    M2 = J[M2[mask]]

    #Form and return sparse matrix
    W = sparse.coo_matrix((np.ones_like(M1), (M1,M2)),shape=(n,n))
    return W.tocsr()

def gamma(h,alpha):
    return (1-alpha)*h*h/alpha

def v(X,h,alpha):
    return 2 - (1+gamma(h,alpha)*np.pi**2/2)*(np.cos(2*np.pi*X[:,0]) + np.cos(2*np.pi*X[:,1]))

def u_true(X):
    return 2 - (np.cos(2*np.pi*X[:,0]) + np.cos(2*np.pi*X[:,1]))

Phi = 1/np.pi

def trial(i,n,h,alpha):
    n = int(n)
    X = np.random.rand(n,2)
    W = Phi*torus_eps_graph(X,h)
    deg = gl.degrees(W)
    conn = gl.isconnected(W)
    u = n*h*h*gl.PageRank(W,alpha=1-alpha,v=v(X,h,alpha),tol=1e-10)/deg
    err = np.max(np.absolute(u-u_true(X)))

    print('%d,%d,%f,%f,%f,%d'%(i,n,h,alpha,err,conn),flush=True)

num_trials = 20

print('exponent,n,h,alpha,error,connected')
for n in [1,2,4,8]:
    n = int(n*1e4)

    for j in range(5):
        h = np.log(n)*n**(-1/2)
        alpha = 30*h*h
        Parallel(n_jobs=num_trials)(delayed(trial)(2,n,h,alpha) for i in range(num_trials))

        h = 2*n**(-1/3)
        alpha = 20*h*h
        Parallel(n_jobs=num_trials)(delayed(trial)(3,n,h,alpha) for i in range(num_trials))

        h = n**(-1/4)
        alpha = 10*h*h
        Parallel(n_jobs=num_trials)(delayed(trial)(4,n,h,alpha) for i in range(num_trials))


#mlab.figure(bgcolor=(1,1,1),size=(800,800))
#mlab.triangular_mesh(X[:,0],X[:,1],u,Tri)

