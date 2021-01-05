#Experiment with using pagerank for data depth and computing median images
import graphlearning as gl 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

#Load MNIST labels and results of k-nearest neighbor search
dataset = 'MNIST'
#dataset = 'FashionMNIST'
X = gl.load_dataset(dataset)
labels = gl.load_labels(dataset)
I,J,D = gl.load_kNN_data(dataset)
X_raw = gl.load_dataset(dataset,metric='raw')

#Build k-NN graph 
k=10
#for asymmetric graph
W = gl.weight_matrix(I,J,D,k,symmetrize=False)
#for symmetric graph
#W = gl.weight_matrix(I,J,D,k,symmetrize=True)
n = W.shape[0]

#Plotting
numw = 16
numh = 10

for alpha in [0.95]:
        print("Alpha:", alpha)
        f_rand, axarr_rand = plt.subplots(numh,numw,gridspec_kw={'wspace':0.1,'hspace':0.1})
        f_pagerank_median, axarr_pagerank = plt.subplots(numh,numw,gridspec_kw={'wspace':0.1,'hspace':0.1})
        f_rand.suptitle('Random Images')
        f_pagerank_median.suptitle('Highest Ranked Images')

        for label in range(10):
            print("Digit %d..." % label)
            # Subset labels
            X_sub = X[labels == label, :]
            X_raw_sub = X_raw[labels == label, :]
            ind_rand = np.random.choice(X_sub.shape[0], numw)
            
            #set v as the characteristic function of the each label (0, 1, ... )
            v = np.zeros(n)
            v[labels==label]=1 #char fun 

            #Run pagerank
            u = gl.PageRank(W,v=v,alpha=alpha)

            #Display highest ranked images
            ind = np.argsort(-u) #indices to sort u

            #Visualization 
            for j in range(numw):
                #pagerank median 
                img = X[ind[j], :]
                m = int(np.sqrt(img.shape[0]))
                img = np.reshape(img, (m, m))
                if dataset == 'MNIST':
                    img = np.transpose(img)
                axarr_pagerank[label, j].imshow(img, cmap='gray')
                axarr_pagerank[label, j].axis('off')
                axarr_pagerank[label, j].set_aspect('equal')

                #random 
                img = X_raw_sub[ind_rand[j], :]
                m = int(np.sqrt(img.shape[0]))
                img = np.reshape(img, (m, m))
                if dataset == 'MNIST':
                    img = np.transpose(img)
                axarr_rand[label, j].imshow(img, cmap='gray')
                axarr_rand[label, j].axis('off')
                axarr_rand[label, j].set_aspect('equal')
        plt.show()

