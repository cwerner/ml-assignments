from __future__ import division

import numpy as np
import sys
import math
from numpy.linalg import inv

from random import randint

X = np.genfromtxt(sys.argv[1], delimiter = ",")

# Number of Iterations
iterations = 10
# Number of clusters = 5 = K
K = 5

N, d = X.shape 


def KMeans(X):
	#perform the algorithm with 5 clusters and 10 iterations...you may try
    #  others for testing purposes, but submit 5 and 10 respectively

    Ci = np.zeros((N, 1))
    Ni = np.zeros(K)
    centroids = np.zeros((K, d))

    for k in range(K):
        centroids[k] = X[randint(0, N-1)]


    for it in range(iterations):

        # expectation step ------------
        
        # reset Ni
        Ni *= 0

        for n in range(N):
            # distances

            # list of distances to each cluster
            distances = []
            for k in range(K):
                # iterate over the dimensions
                # should we also take the sqrt?
                distance = 0.0
                for _d in range(d):
                    distance += (X[n, _d] - centroids[k, _d]) ** 2
                distances.append( distance )

            c = distances.index(min(distances))
            Ci[n] = c
            Ni[c] += 1
        

        # maximization step --------------
        centroids = np.zeros((K,d))

        for n in range(N):
            c = int(Ci[n])
            centroids[c] += X[n] / Ni[c]


        # write state --------------------
        filename = "centroids-" + str(it+1) + ".csv" #"i" would be each iteration
        np.savetxt(filename, centroids, delimiter=",")
    return (centroids, Ci, Ni)


def EMGMM(X, kmeans_centroids, Ci, Ni):

    phi = np.zeros((N, K))
    pi  = np.zeros(K)

    for k in range(K):
        pi[k] = (1.0 / K)
    
    sigmas = [np.zeros((d,d)) for k in range(K)]

    k = 0

    #Ci = np.zeros(N, dtype='int')
    #Ni = np.zeros(K)

    # init centroids (with kmeans centroids)
    centroids = kmeans_centroids

    # new attempt

    for n in range(N):
        y = int(Ci[n][0])
        print(y)
        X_to_c = X[n] - centroids[k]
        X_to_c = X_to_c[np.newaxis]       
        sigmas[y] += np.dot(X_to_c.T, X_to_c)
    
    for k in range(K):
        sigmas[k] /= Ni[k]



    #for k in range(K):
    #    #sigmas[k] /= Ni[k]

    #    sigma = np.zeros((d,d))

    #    for n in range(N):

    #        X_to_c = (X[n] - centroids[k])[np.newaxis]

    #        sigma_N = np.dot(X_to_c.T, X_to_c)
    #        sigma_N *= phi[n, k]

    #        sigma += sigma_N
        
    #    sigma /= (pi[k] * N)
    #sigmas[k] = sigma

    print(sigmas)
    print(centroids)

    # maybe we are missing something here !!!


    for it in range(iterations):
        print("iteration %d ..." % it)

        # expectation step,
        # update phi and pi

        phi = np.zeros((N, K))

        for n in range(N):
            k_sum = 0.0

            for k in range(K):

                det = np.linalg.det(sigmas[k])
                X_minus_mu = (X[n] - centroids[k])[np.newaxis]
                MMul = np.dot(np.dot(X_minus_mu, inv(sigmas[k])), X_minus_mu.T)

                expectation = math.exp(-0.5 * MMul)

                phi[n, k] = (pi[k] * det ** (-0.5)) * expectation

                k_sum += phi[n, k]
            
            # normalize
            phi[n] = phi[n] / k_sum
            pi += phi[n]
        
        pi /= float(N)


        # maximization step

        # reset centroids
        centroids = np.zeros((K, d))

        for n in range(N):
            for k in range(K):
                centroids[k] += ( X[n] * phi[n, k] ) / (pi[k]*N) 

        for k in range(K):
            sigma = np.zeros((d,d))

            for n in range(N):

                X_to_c = (X[n] - centroids[k])[np.newaxis]

                sigma_N = np.dot(X_to_c.T, X_to_c)
                sigma_N *= phi[n, k]

                sigma += sigma_N
        
            sigma /= (pi[k] * N)
        sigmas[k] = sigma


        filename = "pi-" + str(it+1) + ".csv" 
        np.savetxt(filename, pi, delimiter=",") 
        filename = "mu-" + str(it+1) + ".csv"
        np.savetxt(filename, centroids, delimiter=",")  #this must be done at every iteration
    
        for k in range(K): #k is the number of clusters 
            filename = "Sigma-" + str(k+1) + "-" + str(it+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
            np.savetxt(filename, sigmas[k], delimiter=",")


# perform the calculations

centroids, Ci, Ni = KMeans(X)

EMGMM(X, centroids, Ci, Ni)
