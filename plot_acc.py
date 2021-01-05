import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('acc.csv')

dfexp = df['exponent'].values
dfN = df['n'].values
dferr = df['error'].values
dfh = df['h'].values
dfalpha = df['alpha'].values

uni = np.unique(dfN)
m = len(uni)

N = np.zeros((m,3))
err = np.zeros((m,3))
std = np.zeros((m,3))
h = np.zeros((m,3))
alpha = np.zeros((m,3))

j = 0
for exp in [2,3,4]:
    i = 0
    for n in uni:
        I = (dfN==n) & (dfexp == exp)
        N[i,j] = np.mean(dfN[I])
        err[i,j] = np.mean(dferr[I])
        std[i,j] = np.std(dferr[I])
        h[i,j] = np.mean(dfh[I])
        alpha[i,j] = np.mean(dfalpha[I])
        i += 1
    j += 1


plt.ion()
plt.figure()
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 14})
styles = ['^b-','or-','dg-','sk-','pm-','xc-','*y-']


plt.plot(h[:,0],err[:,0],styles[0],label=r'$h = \log(n)n^{-\frac{1}{2}}$')
plt.plot(h[:,1],err[:,1],styles[1],label=r'$h = 2 n^{-\frac{1}{3}}$')
plt.plot(h[:,2],err[:,2],styles[2],label=r'$h = n^{-\frac{1}{4}}$')
plt.xlabel(r'Connectivity radius $h$')
plt.xlim((1.05*np.max(h),0.9*np.min(h)))
#plt.xticks([5,10,15,20])
plt.ylabel(r'$L^\infty$ error')
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(True)
plt.savefig('convergence_test.png')
plt.savefig('convergence_test.eps')
plt.savefig('convergence_test.pdf')
plt.show()

