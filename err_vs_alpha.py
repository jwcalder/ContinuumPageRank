import graphlearning as gl
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

dataset = 'mnist'
metric = 'raw'

#Consturct weight matrix and distance matrix
knn_data = gl.weightmatrix.load_knn_data(dataset,metric=metric)

plt.figure()
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 14})
styles = ['^b-','or-','dg-','sk-','pm-','xc-','*y-']


i=0
for k in [5,10,20]:
    W = gl.weightmatrix.knn(None, k, knn_data=knn_data, symmetrize=False)
    G = gl.graph(W)
    n = G.num_nodes
    knn_dist = np.mean(knn_data[1][:,k-1])

    err_list = []
    knn_dist_list = []
    alpha_list = []

    for alpha in np.arange(0.05,0.55,0.05):
        u = G.page_rank(alpha=1-alpha,tol=1e-10)
        err = np.linalg.norm(n*u - np.ones(n))/np.sqrt(n)
        err = np.max(np.absolute(n*u - np.ones(n)))
        print('k=%d,alpha=%f, err=%f'%(k,alpha,err))
        knn_dist_list.append(knn_dist)
        alpha_list.append(alpha)
        err_list.append(err)

    knn_dist = np.array(knn_dist_list)
    err = np.array(err_list)
    alpha = np.array(alpha_list)
    plt.plot(alpha,err,styles[i],label=r'$k=%d$'%k); i+=1
    p = np.polyfit(np.log(alpha),np.log(err),1)
    print(p)

plt.xlabel(r'Teleportation probability $\alpha$')
#plt.xlim((5,21))
#plt.xticks([5,10,15,20])
plt.ylabel(r'$L^\infty$ distance to teleportation distribution')
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)
plt.show()

