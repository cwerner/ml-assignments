#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# hw3_clustering.py
# =================
#
# Instructions:
#  In this assignment you will implement the K-means and EM Gaussian
#  mixture models. We will give you n data points {x1,...,xn} where
#  each xi E Rd.
#  Recall that with K-means we are trying to find K centroids
#  {mu1,...,muK} and the corresponding assignments of each data point
#  {c1,...,cn}, where each ci E {1,...,K} and ci indicates which of
#  the K clusters the observation xi belongs to. The objective function
#  that we seek to minimize can be written
#
#  L = sum(i,1...n)  sum(k,1...K) l(ci=k)||xi-muk||^2
#
#  We also discussed using the EM algorithm to learn the parameters
#  of a Gaussian mixture model. For this model, we assume a
#  generative process for the data as follows,
#
#  xi|ci ~ Normal(muc, sumci), ci ~ Discrete(pi).
#
#  In other words, the ith observation is first assigned to one of K
#  clusters according to the probabilities in vector π, and the value
#  of observation xi is then generated from one of K multivariate
#  Gaussian distributions, using the mean and covariance indexed by ci
#
#  The EM algorithm discussed in class seeks to maximize
#
#  p(x1,...,xn|pi, mu, sum) = Prod(i=1...n) p(xi|pi, mu, sum)
# 
#  over all parameters pi, mu1,...,muK, sum1, ..., sumK using the
#  cluster assignments c1,...,cn as the hidden data.
#
#  OUTPUT: 
#  You should write your K-means and EM-GMM codes to learn 5 clusters.
#  Run both algorithms for 10 iterations. You can initialize your
#  algorithms arbitrarily. We recommend that you initialize the K-means
#  centroids by randomly selecting 5 data points. For the EM-GMM, we
#  also recommend you initialize the mean vectors in the same way, and
#  initialize π to be the uniform distribution and each Σk to be the
#  identity matrix. 
#
# When executed, you will have your code write several output files each
# described below. It is required that you follow the formatting
# instructions given below. Where you see [iteration] and [cluster]
# below, replace these with the iteration number and the cluster number.
#
# centroids-[iteration].csv: This is a comma separated file containing
#  the K-means centroids for a particular iteration. The kth row should
#  contain the kth centroid, and there should be 5 rows. There should be
#  10 total files. For example, "centroids-3.csv" will contain the
#  centroids after the 3rd iteration.
#
# pi-[iteration].csv: This is a comma separated file containing the
#  cluster probabilities of the EM-GMM model. The kth row should contain
#  the kth probability, πk, and there should be 5 rows. There should be
#  10 total files. For example, "pi-3.csv" will contain the cluster
#  probabilities after the 3rd iteration.
#
# mu-[iteration].csv: This is a comma separated file containing the means
#  of each Gaussian of the EM-GMM model. The kth row should contain the 
#  kth mean , and there should be 5 rows. There should be 10 total files.
#  For example, "mu-3.csv" will contain the means of each Gaussian after
#  the 3rd iteration.
#
# Sigma-[cluster]-[iteration].csv: This is a comma separated file
#  containing the covariance matrix of one Gaussian of the EM-GMM model.
#  If the data is d-dimensional, there should be d rows with d entries
#  in each row. There should be 50 total files. For example, 
#  "Sigma-2-3.csv" will contain the covariance matrix of the 2nd
#  Gaussian after the 3rd iteration.
#
#


import numpy as np
import pandas as pd
import scipy as sp
import sys

from scipy import stats

# seed
seed = 123456789
np.random.seed(seed)

# input data
X = np.genfromtxt(sys.argv[1], delimiter = ",")

# constants
# n number of samples, d = dimensionality
N, d = X.shape
K = 5				# number of clusters
iterations = 10		# number of iteratins

# turn plotting on/ off
PLOT = True

# functions -------------------------------------

def create_plot(X, mu, c, iteration=0):
	""" Optional, simple plotting code to check what we are doing
	"""
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt

	# cluster colors
	Dcol = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'cyan'} 

	if d >= 3:
		if d>3:
			print('Using only first 3 dimensions for plotting...')
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		if iteration == 0:
			ax.scatter(X[:,0], X[:, 1], X[:,2], c='black', s=5, alpha=.5)	
		else:
			ax.scatter(X[:,0], X[:, 1], X[:,2], c=[Dcol[_] for _ in c], marker='+', s=5, alpha=.5)
		ax.scatter(mu[:,0], mu[:, 1], mu[:,2], c=[Dcol[_] for _ in range(len(mu))], marker='P', s=25, alpha=1)
		fig.savefig('kmeans_3d_%02d.pdf' % iteration)
	elif d == 2:
		fig = plt.figure()
		ax = fig.add_subplot(111)

		# inital color for datapoints: black
		if iteration == 0:
			ax.scatter(X[:,0], X[:, 1], c='black', s=5, alpha=.5)	
		else:
			ax.scatter(X[:,0], X[:, 1], c=[Dcol[_] for _ in c], marker='+', s=5, alpha=.5)
		ax.scatter(mu[:,0], mu[:, 1], c=[Dcol[_] for _ in range(len(mu))], marker='P', s=25, alpha=1)
		fig.savefig('kmeans_2d_%02d.pdf' % iteration)		
	else:
		print('Only one dimension recognized. Abort.')
		sys.exit()

def random_init_centroids(X):
	""" Take input and choose random centroids
	"""
	mu = []
	x_min = np.min(X, axis=0)
	x_max = np.max(X, axis=0)
	for _ in range(K): 
		k_init = (np.random.random(d) * (x_max - x_min)) + x_min
		mu.append(k_init)

	return np.array(mu)

# coordinate descent
def kmeans_assignment_step(X_, mu):
	""" K-Means assigment step
	"""

	c = np.zeros(X.shape[0]) 

	# TODO: Check if we need to normalize for full scores?
	#X_ = stats.zscore(X, axis=0)

	# iterate over the data points (z-score normalized)
	for ix, xi in enumerate(X_):
		distances = np.zeros(5)
		for kx, mu_k in enumerate(mu):
			distances[kx] = np.sqrt(np.sum(np.power(xi - mu_k, 2)))
		c[ix] = np.argmin(distances)
	return c 

def kmeans_update_step(X_, c, mu):
	""" Update centroids
	""" 

	# TODO: Check if we need to normalize for full scores?
	#X_ = stats.zscore(X, axis=0)

	for k in range(K):
		selection = c == k
		X_sel = X_[selection,:]

		if len(X_sel) > 0:	 
			new_mu_k = np.mean(X_sel, axis=0)
			mu[k] = new_mu_k
	return mu



def KMeans(data):
	# perform the algorithm with 5 clusters and 10 iterations...

	# random initialization for start
	mu = random_init_centroids(X)
	c = np.zeros(X.shape[0]) 

	if PLOT: create_plot(X, mu, c, iteration=0)

	# k-means run (1...10)
	for i in range(iterations):
		# step a - assign to closest centroid
		c = kmeans_assignment_step(X, mu)	
		
		if PLOT: create_plot(X, mu, c, iteration=i+1)

		filename = "centroids-" + str(i+1) + ".csv" 		#"i" would be each iteration
		np.savetxt(filename, mu, delimiter=",")

		# TODO: Check if we need to move this upwards to get full score
		mu = kmeans_update_step(X, c, mu)



  
def EMGMM(data):
	#
	# EM-GMM


	for i in range(iterations):


		# data output
		filename = "pi-" + str(i+1) + ".csv" 
		np.savetxt(filename, pi, delimiter=",") 
		filename = "mu-" + str(i+1) + ".csv"
		np.savetxt(filename, mu, delimiter=",")
    
		for j in range(K): #k is the number of clusters 
		    filename = "Sigma-" + str(j+1) + "-" + str(i+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
		    np.savetxt(filename, sigma[j], delimiter=",")


# PART 1 : K-Means
KMeans(X)


