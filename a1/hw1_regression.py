# HOMEWORK 1
# ==========
#
# Call: $ python hw1_regression.py lambda sigma2 X_train.csv y_train.csv X_test.csv
# 
# NOTE: The values of lambda and sigma2 will be input as strings. You must convert
#  them to a number for your code to work. All numbers should be double-precision
#  floating-point format. 
# 
# Infiles: The csv files that we will input into your code are formatted as follows:
#  X_train.csv: A comma separated file containing the covariates. Each row corresponds
#               to a single vector xi. The last dimension has already been set equal
#               to 1 for all data.
#  y_train.csv: A file containing the outputs. Each row has a single number and the
#               i-th row of this file combined with the i-th row of "X_train.csv"
#               constitutes the training pair (yi,xi).
#  X_test.csv: This file follows exactly the same format as "X_train.csv". No response
#              file is given for the testing data. 
# 
#
# Example output:
#  For example, if λ=2 and σ2=3, then the files you create will be named "wRR_2.csv"
#  and "active_2_3.csv". If your code then learns that w = [3.2; -3.6; 1.4; -0.7],
#  then wRR_2.csv should look like:
#  3.2
#  -3.6
#  1.4
#  -0.7
#  If the first 10 index values you would choose to measure are 724, 12, 109, 42,
#  23, 96, 342, 594, 123, 414, then active_2_3.csv should look like:
# 724,12,109,42,23,96,342,594,123,414
# 
import numpy as np
import sys

PLOT = False
if PLOT:
    import matplotlib.pyplot as plt


# input
lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

# additional info
# Get the number of columns -> dimension of the input vector -> size of identity_matrix
N, d = X_train.shape

## Solution for Part 1
def part1(X, y, lmbda):
    """ Ridge Regression Calulation (Part1)
    """
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    
    # identiy matrix/ could also use np.eye
    I = np.identity(X.shape[1])

    # intermediate dot products
    xTx = np.dot(X.T, X)
    xTy = np.dot(X.T, y)
    
    # calculate the weights
    wRR = np.dot(np.linalg.inv(xTx + lmbda*I), xTy) 
    
    return wRR

wRR = part1(X_train, y_train, lambda_input)
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n")


## Solution for Part 2
def calc_sigma0s(sigma_prior, X_test, values):
    """ Calculate Sigma0
    """
    sigma0s = np.zeros(X_test.shape[0])
    for ix, row in enumerate(X_test):
        # check if these dot products are really necessary
        # ignore values (previously selected values)
        if ix not in values:
            sigma0s[ix] = sigma2_input + np.dot(np.dot(row.T, sigma_prior), row) 
    return sigma0s

def calc_posterior(sigma_prior, row):
    """ Calclate Posterior
    """
    return np.linalg.inv(
        (1.0 / sigma2_input * row) * row.T + np.linalg.inv(sigma_prior)
    )


def calc_entropy(H_prior, row, sigma_prior):
    """ Calculate Entropy
    """
    return (
        H_prior
        - np.log(1 + 1 / sigma2_input * np.dot(np.dot(row.T, sigma_prior), row))
        * d
        * 0.5
    )


def part2(X, X_test, lmbda, sigma2_input):
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file

    # list of entropies
    entropies = [0.0]

    I = np.identity(X.shape[1])
    xTx = np.dot(X.T, X)

    # define sigma_prior, H_prior
    sigma_prior = np.linalg.inv((lmbda * I) + (1.0/sigma2_input) * xTx)
    H_prior = entropies[-1]

    # do the loop
    test_data = X_test[:]
    
    # active/ result values (also used to skip rows in calc_sigma0s)
    values = []

    # pull ten points with greatest sigma
    for _ in range(10):
        sigma0_ = calc_sigma0s(sigma_prior, test_data, values)

        # get the maximum sigma
        max_sigma0 = np.argmax(sigma0_)
        values.append(max_sigma0)

        # calculate entropy
        H_post  = calc_entropy(H_prior, test_data[max_sigma0,:], sigma_prior)
        entropies.append(H_post)

        # calculate posterior
        sigma_post = calc_posterior(sigma_prior, test_data[max_sigma0,:])
        
        # update priors for next round
        sigma_prior = sigma_post
        H_prior = H_post

    if PLOT:
        # optional plot of entropy progress
        fig, ax = plt.subplots()
        ax.plot(range(len(entropies)), entropies, 'b-')
        fig.savefig('entropies.pdf')

    return values


active = part2(X_train, X_test, lambda_input, sigma2_input)  # Assuming active is returned from the function
if len(active) > 10:
    active = active[:10]

# increase the indices by 1
active = np.array(active).reshape((1, len(active))) + 1
print(active)
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, fmt='%d', delimiter=",")