from __future__ import division
import sys
import numpy as np
import math


# execute with
# python hw2_classification.py X_train.csv y_train.csv X_test.csv

# input:
#
# The csv files that we will input into your code are formatted as follows:.
#   X_train.csv: A comma separated file containing the covariates. 
#      Each row corresponds to a single vector xi.
#   y_train.csv: A file containing the classes. Each row has a single number
#      and the i-th row of this file combined with the i-th row of "X_train.csv"
#      constitutes the labeled pair (yi, xi). There are 10 classes having index
#      values 0,1,2,...,9.
#   X_test.csv: This file follows exactly the same format as "X_train.csv". No
#      class file is given for the testing data.
#
#
# output:
#
# probs_test.csv: This is a comma separated file containing
#    the posterior probabilities of the label of each row in
#    "X_test.csv". Since there are 10 classes, the i-th row
#    of this file should contain 10 numbers, where the j-th
#    number is the probability that the i-th testing point
#    belongs to class j-1 (since classes are indexed 0 to 9 here).



# ten classes (indexed 0...9)
K = 10


def main(X_train, y_train, X_test):

    # n = observations
    # i = number of parameters
    n, i = X_train.shape

    print("n:", n)
    print("i:", i)


    n_out = X_test.shape[0]


    mu = np.zeros((K,i))
    sigma = [np.zeros((i,i)) for k in range(K)]

    # predictions
    pred = np.zeros(shape=(n_out, K))

    # intialize pi with label counts from data
    pi = np.zeros(K, dtype='float')

    # iterate over data and count occurrances
    for sample in range(n):
        label = int(y_train[sample])
        pi[label] += 1
        mu[label] += X_train[sample]

    # iterate over clusters and scale
    for k in range(K):
        pi[k] /= float(n)
        if float(n)*pi[k] > 0:
            mu[k] /= (float(n)*pi[k])

    #
    # 
    print("Pi : ")
    print(pi)

    print("Mu : ")
    print(mu)

    print("Sigma : ")
    print(sigma)

    #  
    for sample in range(n):
        x_i = X_train[sample]
        label   = int(y_train[sample])

        # init this as 2d so we can transpose
        V = np.zeros((i,1))

        for _i in range(i):
            V[_i] = x_i[_i] - mu[label][_i]

        H = V.T

        dprod = np.dot(V, H)

        #for _i in range(i):
        sigma[label] += dprod #[_i] 


    print("Sigma : ")
    print(sigma)


    selector = 0

    # rewrite, check that this works

    print(len(sigma))
    print(len(pi))
    # update sigma
    for _i in range(i): #sigma.shape[0]):
        #if _ == i:
        #    selector += 1
        
        sigma[_i] /= n * pi[_i]


    print("Sigma : ")
    print(sigma)


    # final algo
    for sample_p in range(n_out):

        _sum = 0

        for k in range(K):

            for _i in range(i):
            #
                sigma_l = sigma[_i]

            #    # rewrite this to a list of sigmas (!)
            #    sigma_l[_] = sigma[i*k+_]
            
                pred[sample_p, k] = pi[k] * np.linalg.det(sigma_l) ** (-0.5)

            V = np.zeros((i, 1))

            for _i in range(i):
                V[_i] = X_test[sample_p, _i] - mu[k, _i]
            
            H = V.T

            for _i in range(i):
            #
                sigma_l = sigma[_i]
                pred[sample_p, k] *= math.exp(-0.5* np.dot(H, np.dot(sigma_l, V)))

            _sum += pred[sample_p, k]

        for k in range(K):
            pred[sample_p, k] *= 1/ _sum


    # assuming final_outputs is returned from function
    np.savetxt("probs_test.csv", pred, delimiter=",") 
    # write output to file

if __name__ == '__main__':

    # load data
    X_train = np.genfromtxt(sys.argv[1], delimiter=",")
    y_train = np.genfromtxt(sys.argv[2])
    X_test = np.genfromtxt(sys.argv[3], delimiter=",")

    main(X_train, y_train, X_test)