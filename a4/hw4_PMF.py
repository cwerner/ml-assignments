from __future__ import division
import numpy as np
from numpy.linalg import inv, norm
import sys

train_data = np.genfromtxt(sys.argv[1], delimiter = ",")

lam = 2
sigma2 = 0.1
d = 5

iterations = 50

# Implement function here
def PMF(data):

    N = data.shape[0]

    N1 = 0  # users
    N2 = 0  # items

    # count
    for n in range(N):
        if data[n, 0] > N1:
            N1 = int(data[n][0])
        if data[n, 1] > N2:
            N2 = int(data[n][1]) 

    print(N1, N2)

    # populate Omega_ij

    omega_ij = np.zeros((N1, N2))

    for n in range(N):
        omega_ij[int(data[n,0])-1, int(data[n,1])-1] = int(data[n,2])
    
    # factorized matices
    U = np.zeros((N1, d))
    V = np.zeros((d, N2)) 

    # init matrices
    # U, V with N(0, lamba-1*I)
    for _d in range(d):
        for n in range(N2):
            V[_d, n] = np.random.normal(0, 1.0 / lam)

    for _d in range(d):
        for n in range(N1):
            U[n, _d] = np.random.normal(0, 1.0 / lam)

    # MAP function (that we want to minimize)

    # valid for both parts
    I = np.identity(d)
    I_L_S = I * lam * sigma2

    # iterate
    U_matrices = []
    V_matrices = []
    L = []
    for it in range(iterations):
        print("iteration %d..." % it)

        # update user
        for i in range(N1):

            SUM = np.zeros((d,d))
            Mij_vj = np.zeros((d,1))

            for j in range(N2):
                # if we have a rating
                if omega_ij[i,j] > 0:
                    # check if we can rewrite this, ugly
                    Vj = V[:,j][np.newaxis]
                    SUM += np.dot(Vj.T, Vj)
                    
                    # delete:
                    #Mij_vj += numpy.transpose(numpy.multiply(Vj,OmegaIJ[i][j]))
                    Mij_vj += (Vj * omega_ij[i,j]).T

            U[i] = np.dot(inv(I_L_S + SUM), Mij_vj).T

        # update object
        for j in range(N2):

            SUM = np.zeros((d,d))
            Mij_ui = np.zeros((d,1))

            for i in range(N1):
                if omega_ij[i,j] > 0:
                    Ui = U[i][np.newaxis]
                    SUM += np.dot(Ui.T, Ui)

                    Mij_ui += (Ui * omega_ij[i,j]).T 

            V[:,j] = np.dot(inv(I_L_S + SUM), Mij_ui).T
        
        # calculate L 

        # reset helper
        SUM_Mij_uivj, SUM_ui, SUM_vj = 0,0,0
        for i in range(N1):
            for j in range(N2):
                if omega_ij[i,j] > 0:
                    Ui = U[i]
                    Vj = V[:,j][np.newaxis]
                    Vj_b = V[:,j]
                    UV = np.dot(Ui, Vj.T)

                    SUM_Mij_uivj += (omega_ij[i,j] - UV) ** 2 / (2.0 * sigma2)
                    SUM_ui += norm(Ui, ord=2) * lam / 2.0
                    SUM_vj += norm(Vj_b, ord=2) * lam / 2.0

        L.append( int( -SUM_Mij_uivj - SUM_ui - SUM_vj ))
        # store matrices
        U_matrices.append(U)
        V_matrices.append(V.T)

    return (L, U_matrices, V_matrices)


# Assuming the PMF function returns Loss L, U_matrices and V_matrices
# (refer to lecture)
L, U_matrices, V_matrices = PMF(train_data)

np.savetxt("objective.csv", L, delimiter=",")

np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("V-50.csv", V_matrices[49], delimiter=",")
